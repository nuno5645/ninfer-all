#include "core/expert_delivery.h"

#include "core/cpu_spin_pool.h"
#include "core/cuda_host_handshake.h"
#include "core/cuda_host_handshake.cuh"
#include "core/device.h"
#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/expert_direct.h"
#include "core/expert_mirror.h"
#include "core/expert_split.h"
#include "core/file_read_queue.h"
#include "core/growable_region.h"
#include "core/pinned_memory.h"
#include "core/vmm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

namespace ninfer {
namespace {

// A round's copies a call's post work issues at most: experts, and bytes read from files.
constexpr std::size_t kRoundExperts = 8;
constexpr std::uint64_t kRoundFileBytes = 8ULL << 20;

// needed[e] of a call: its pairs routed to expert e, and whether the table entry the split read
// lies in the persistent cache.
constexpr std::uint32_t kCountMask = (1u << 29) - 1;
// Request flag: begin() received the family's CPU inputs, so the call may leave misses to the CPU.
constexpr std::uint32_t kCpuCall = 1;
constexpr std::uint32_t kHit     = 1u << 30;
constexpr std::uint32_t kMiss    = 1u << 31;
// hp-tune (miss-skip): needed[e] of an expert the split dropped (its count reads zero).
constexpr std::uint32_t kSkip    = 1u << 29;
// The split's shared count of an expert whose table entry hits; with miss-skip, of a missing
// expert one of whose pairs must stay (kKeep), and of an expert it drops (kDrop).
constexpr std::uint32_t kInCache = 1u << 31;
constexpr std::uint32_t kKeep    = 1u << 30;
constexpr std::uint32_t kDrop    = 1u << 29;

// hp-tune (miss-skip): the split's view of ExpertDelivery::Skip and the environment's limits.
struct SplitSkip {
    float* weights     = nullptr; // null: no miss-skip
    const float* scale = nullptr;
    int top_k          = 0;
    int rank           = 1 << 30;
    float share        = 0.0f;
    int renorm         = 0;
};

std::uint64_t aligned(std::uint64_t n) {
    if (n > std::numeric_limits<std::uint64_t>::max() - 255) {
        throw std::invalid_argument("expert delivery: byte extent overflow");
    }
    return (n + 255) / 256 * 256;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a * b;
}

// Use the core's existing fixed-address allocation when the Program supports model suspend. An
// elastic cache is a growable region: addresses for every expert, `bytes` of them in slots.
struct Storage {
    DeviceBuffer owned;
    VmmRegion region;
    GrowableRegion growable;
    void* p = nullptr;
    std::size_t bytes = 0;
    Storage() = default;
    Storage(std::size_t n, int device, bool suspendable) : bytes(n) {
        if (suspendable) {
            region = VmmRegion(n, {.device = device});
            p = region.span().data;
        } else {
            owned = DeviceBuffer(n);
            p = owned.p;
        }
    }
    void fill() {
        if (growable) {
            // An elastic cache maps a run of granules for each of its pools: zero every run.
            const std::size_t g = growable.granule();
            for (std::size_t at = 0; at < growable.bytes();) {
                if (!growable.mapped(at, g)) {
                    at += g;
                    continue;
                }
                std::size_t end = at + g;
                while (end < growable.bytes() && growable.mapped(end, g)) { end += g; }
                CUDA_CHECK(ninfer::fill_and_wait(static_cast<std::byte*>(p) + at, 0, end - at));
                at = end;
            }
        } else {
            CUDA_CHECK(ninfer::fill_and_wait(p, 0, bytes));
        }
    }
    [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
        return growable ? growable.mapped_bytes() : region ? region.reserved_bytes() : bytes;
    }
    void release_backing() {
        if (growable) { growable.release_backing(); }
        else { region.release_backing(); }
    }
    void restore_backing() {
        if (growable) { growable.restore_backing(); }
        else { region.restore_backing(); }
    }
};

// Device ranges a call hands the host service (the CPU's operands), into mapped host memory: the
// pairs' expert ids, then up to kCpuInputs inputs of the family's.
constexpr int kCpuInputs = 3;
struct HostCopies {
    const std::byte* source[1 + kCpuInputs]{};
    std::byte* target[1 + kCpuInputs]{};
    std::uint32_t bytes[1 + kCpuInputs]{};
};

// One block: a shared histogram of the call's experts, then each pair's cached/missing id and each
// set's experts (their count, then the experts ascending) by the table entry the expert has now.
// It writes the mask and `copies` into mapped host memory and publishes the request itself.
// Tables change only at missing entries or at experts absent from this call, so cached consumers
// can read them while the host prepares the rest. An entry in a lent bank (`lent`: [bank][low,
// high), empty when not lent) hits in a narrow call only: a wide call, which routes half the
// layer's experts or more, may take that bank back while its cached pairs run.
__global__ void split_routes(const std::int32_t* ids, std::uint32_t pairs, int experts,
                             const void* const* tables, std::uintptr_t low, std::uintptr_t high,
                             const std::uintptr_t* lent, std::int32_t* cached,
                             std::int32_t* missing, std::int32_t* cached_experts,
                             std::int32_t* missing_experts, std::uint32_t* needed,
                             HostCopies copies, handshake::DeviceMailbox mailbox, std::uint32_t tag,
                             std::uint32_t flags, SplitSkip skip) {
    extern __shared__ std::uint32_t counts[];
    __shared__ std::uint32_t routes, sets[2][32], listed[2];
    if (threadIdx.x == 0) {
        routes    = 0;
        listed[0] = listed[1] = 0;
    }
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { counts[e] = 0; }
    __syncthreads();
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { atomicAdd(counts + e, 1u); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        if (counts[e]) { atomicAdd(&routes, 1u); }
    }
    __syncthreads();
    const bool wide = 2 * routes >= std::uint32_t(experts);
    const std::uintptr_t lent_low[2]{lent[0], lent[2]}, lent_high[2]{lent[1], lent[3]};
    const auto hit = [&](const void* entry) {
        const auto address = reinterpret_cast<std::uintptr_t>(entry);
        if (address >= low && address < high) { return true; }
        return !wide && ((address >= lent_low[0] && address < lent_high[0]) ||
                         (address >= lent_low[1] && address < lent_high[1]));
    };
    // hp-tune (miss-skip), narrow calls only: the hits first, then each missing pair whose rank in
    // its row (by probability, 0 the largest) is below skip.rank and whose share of the row's
    // probability is at least skip.share keeps its expert. Padding pairs (-1) count for nothing.
    const bool skipping = skip.weights != nullptr && !wide;
    const auto probability = [&](std::uint32_t j) {
        const int x = ids[j];
        if (x < 0 || x >= experts) { return 0.0f; }
        return skip.scale ? skip.weights[j] / skip.scale[x] : skip.weights[j];
    };
    if (skipping) {
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            if (counts[e] && hit(tables[e])) { counts[e] |= kInCache; }
        }
        __syncthreads();
        for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
            const int e = ids[p];
            if (e < 0 || e >= experts || (counts[e] & kInCache)) { continue; }
            const std::uint32_t base = p / std::uint32_t(skip.top_k) * std::uint32_t(skip.top_k);
            const float w = probability(p);
            float sum = 0.0f;
            int rank  = 0;
            for (std::uint32_t j = base; j < base + std::uint32_t(skip.top_k); ++j) {
                const float v = probability(j);
                sum += v;
                if (j != p && (v > w || (v == w && j < p))) { ++rank; }
            }
            const bool drop = rank >= skip.rank || w < skip.share * sum;
            if (!drop) { atomicOr(counts + e, kKeep); }
        }
        __syncthreads();
    }
    // Each warp ballots its experts of a block-wide pass; earlier warps and passes place them.
    // A routed expert's count keeps its hit in the top bit for the pairs below.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int first = 0; first < experts; first += blockDim.x) {
        const int e               = first + threadIdx.x;
        const std::uint32_t count = e < experts ? counts[e] : 0;
        const bool dropped = skipping && count != 0 && !(count & kInCache) && !(count & kKeep);
        const bool routed  = count != 0 && !dropped;
        const bool in_cache =
            routed && (skipping ? (count & kInCache) != 0 : hit(tables[e]));
        const unsigned hits       = __ballot_sync(~0u, in_cache);
        const unsigned misses     = __ballot_sync(~0u, routed && !in_cache);
        if (e < experts) {
            needed[e] = dropped  ? kSkip
                        : !routed ? 0u
                                  : (count & kCountMask) | (in_cache ? kHit : kMiss);
            if (in_cache) { counts[e] = count | kInCache; }
            if (dropped) { counts[e] = count | kDrop; }
        }
        if (lane == 0) {
            sets[0][warp] = __popc(hits);
            sets[1][warp] = __popc(misses);
        }
        __syncthreads();
        std::uint32_t at[2]{listed[0], listed[1]};
        for (int w = 0; w < warp; ++w) {
            at[0] += sets[0][w];
            at[1] += sets[1][w];
        }
        const unsigned below = (1u << lane) - 1;
        if (routed) {
            if (in_cache) {
                cached_experts[1 + at[0] + __popc(hits & below)] = e;
            } else {
                missing_experts[1 + at[1] + __popc(misses & below)] = e;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < int(blockDim.x >> 5); ++w) {
                listed[0] += sets[0][w];
                listed[1] += sets[1][w];
            }
        }
        __syncthreads();
    }
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e            = ids[p];
        const bool valid       = e >= 0 && e < experts;
        const bool cached_pair = valid && (counts[e] & kInCache) != 0;
        const bool dropped     = valid && (counts[e] & kDrop) != 0;
        cached[p]  = cached_pair ? e : -1;
        missing[p] = valid && !cached_pair && !dropped ? e : -1;
    }
    // hp-tune (miss-skip): a dropped pair weighs zero; with renorm the row's kept weights scale
    // up to the row's probability. Before the copies, which carry the weights to the CPU.
    if (skipping) {
        const std::uint32_t k = std::uint32_t(skip.top_k);
        for (std::uint32_t row = threadIdx.x; row < pairs / k; row += blockDim.x) {
            float sum = 0.0f, kept = 0.0f;
            bool any  = false;
            for (std::uint32_t j = row * k; j < row * k + k; ++j) {
                const int e = ids[j];
                if (e < 0 || e >= experts) { continue; }
                const float v = probability(j);
                sum += v;
                if (counts[e] & kDrop) {
                    any = true;
                } else {
                    kept += v;
                }
            }
            if (!any) { continue; }
            const float factor = skip.renorm && kept > 0.0f ? sum / kept : 1.0f;
            for (std::uint32_t j = row * k; j < row * k + k; ++j) {
                const int e = ids[j];
                if (e < 0 || e >= experts) { continue; }
                skip.weights[j] = (counts[e] & kDrop) ? 0.0f : skip.weights[j] * factor;
            }
        }
        __syncthreads();
    }
    // 16-byte words where both ends and the length allow, else 4-byte words or bytes: a warp's
    // writes into mapped memory cross the bus as fewer, wider writes.
    for (int c = 0; c < 1 + kCpuInputs; ++c) {
        const std::byte* source = copies.source[c];
        std::byte* target       = copies.target[c];
        const std::uint32_t n   = copies.bytes[c];
        const auto ends = reinterpret_cast<std::uintptr_t>(source) |
                          reinterpret_cast<std::uintptr_t>(target) | n;
        if ((ends & 15) == 0) {
            for (std::uint32_t i = threadIdx.x; i < n / 16; i += blockDim.x) {
                reinterpret_cast<uint4*>(target)[i] = reinterpret_cast<const uint4*>(source)[i];
            }
        } else if ((ends & 3) == 0) {
            for (std::uint32_t i = threadIdx.x; i < n / 4; i += blockDim.x) {
                reinterpret_cast<std::uint32_t*>(target)[i] =
                    reinterpret_cast<const std::uint32_t*>(source)[i];
            }
        } else {
            for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) { target[i] = source[i]; }
        }
    }
    // The host reads the mask and the copies once it sees the request.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        cached_experts[0]  = std::int32_t(listed[0]);
        missing_experts[0] = std::int32_t(listed[1]);
        handshake::publish(mailbox, tag, pairs, flags);
    }
}

} // namespace

struct ExpertDelivery::Impl {
    struct Layer {
        ExpertDeliveryLayer source;
        Storage tables;
        std::unique_ptr<PinnedHostBuffer> mirror;
        // Whole-table uploads alternate between two pinned copies, so the mirror can change while
        // an earlier upload still reads its copy.
        std::unique_ptr<PinnedHostBuffer> upload[2];
        cudaEvent_t uploaded[2]{};
        int next_upload = 0;
        bool dirty      = false;
        std::vector<std::int32_t> slot_of;
        std::uint32_t cached = 0; // slots holding this layer's experts
        std::uint32_t pool   = 0; // the pool of slots its experts fit
        // Bank stride of each blob: the blob's bytes when every expert's are equal (the resident
        // layout, one copy per run of experts), else its aligned maximum.
        std::vector<std::uint64_t> stride;
        // The other layer whose wide call followed this layer's last wide call (-1: none yet).
        std::int64_t successor = -1;
        std::vector<std::uint64_t> total; // the narrow calls' routes of each expert
        bool cpu = false;                 // the family computes its experts on the CPU
        // Where each expert's entries point besides a cache slot: the lent slot holding it, and
        // the bank whose layout they address (-1: none).
        std::vector<std::int32_t> lent_of;
        std::vector<std::int8_t> bank_of;
    };
    struct Bank {
        std::int64_t layer = -1;
        std::vector<unsigned char> valid;
    };
    struct Slot {
        std::int32_t layer = -1, expert = -1;
        std::uint32_t pool = 0;
        // Its table entries still point elsewhere: a promotion's are patched at the next call, a
        // round's once its copy has landed.
        bool pending = false;
        std::vector<std::uint64_t> written;
    };
    // A slot of a lent bank: bank k holds slots [k * per_bank, (k + 1) * per_bank). `used`: the
    // last call that routed its expert.
    struct LentSlot {
        std::int32_t layer = -1, expert = -1;
        std::uint64_t used = 0;
        std::vector<std::uint64_t> written;
        // hp-tune (N11''): a read-ahead copy of (layer, expert) is in flight into it; no table
        // entry points at it yet.
        bool ahead = false;
    };

    // The cache slots of the layers whose experts' blobs need the same room (each blob's largest
    // bytes and tail in those layers, 256-byte aligned): the tier's pools (core/expert_tier.h). A
    // pool's slots are ids [first, first + capacity), the first `count` of them backed, at `base`
    // in the cache one after another; an elastic cache reserves `span` bytes for each, a whole
    // number of granules.
    struct Pool {
        std::vector<std::uint64_t> offset, size; // per blob, within a slot
        std::uint64_t slot_bytes = 0, base = 0, span = 0;
        std::int32_t first     = 0;
        std::uint32_t capacity = 0, count = 0, layers = 0;
        std::vector<std::int32_t> free; // popped lowest first
    };
    struct Deferred {
        std::size_t layer;
        std::int32_t expert, slot;
        cudaEvent_t landed = nullptr; // a round's copy; null: a promotion, ordered on `copies`
    };
    // A host-to-device copy of contiguous bytes from one pinned range or a run of file extents.
    struct Piece {
        std::byte* target = nullptr;
        const std::byte* pinned = nullptr;
        std::uintptr_t region = 0;
        std::vector<ExpertExtent> extents;
        std::uint64_t bytes = 0;
    };

    RankContext rank;
    ExpertDeliveryOptions options;
    std::vector<Layer> layers;
    std::uint32_t max_experts = 0;
    // Per blob: a lent slot's offset/size (aligned for the largest blob and tail of any layer), the
    // bank region offset/size and the zero tail kept after a bank run.
    std::vector<std::uint64_t> slot_offset, slot_size, bank_offset, bank_size, tail;
    std::uint64_t slot_bytes = 0, bank_bytes = 0;
    // hp-tune (N11/N12): this token's narrow visits per layer, each loop layer's experts that its
    // last visit routed, the tokens seen, and the pass-to-pass counts by transition (1->2, 2->3).
    struct LoopCounts {
        std::uint64_t visits = 0, routes = 0, previous = 0, repeat = 0, repeat_cache = 0,
                      repeat_lent = 0, repeat_cpu = 0, repeat_copied = 0, fresh_cpu = 0,
                      fresh_copied = 0;
    };
    std::vector<std::uint32_t> loop_visits;
    std::vector<std::vector<unsigned char>> loop_prev;
    std::uint64_t loop_tokens = 0, loop_forced = 0;
    std::int64_t loop_visit = -1; // the served call's visit of its loop layer this token
    LoopCounts loop_counts[3];
    // hp-tune (probe): the served calls' ring (probe_count records published so far) and the
    // record of the call being served.
    static constexpr std::size_t kProbeRing = 1024;
    bool probe_on = false;
    std::vector<ExpertProbeRecord> probe_ring;
    std::atomic<std::uint64_t> probe_count{0};
    ExpertProbeRecord probe_now;
    static std::int64_t probe_clock(std::chrono::steady_clock::time_point at) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count();
    }
    static std::int64_t probe_clock() { return probe_clock(std::chrono::steady_clock::now()); }
    // hp-tune (N11''): the guess area (mapped host memory, per layer 1 + guess_rows * guess_width
    // words), its device address, and the counts (under `mutex`).
    std::unique_ptr<PinnedHostBuffer> guess_host;
    void* guess_mapped        = nullptr;
    std::uint32_t guess_rows  = 0, guess_width = 0;
    ExpertGuessStats guess_counts;
    std::vector<unsigned char> guess_seen;
    bool guess_score = true;

    // hp-tune (N11'', NINFER_INK_PREFETCH): read-ahead copies (on ahead_stream) of a guessed
    // layer's experts into a cache slot (pending) or a lent slot (`ahead`), each with its event;
    // judged once the guessed layer's next narrow call has been served. Counts under `mutex`.
    struct Ahead {
        std::size_t layer   = 0;
        std::int32_t expert = -1, slot = -1;
        bool lent = false, judged = false;
        cudaEvent_t landed = nullptr;
    };
    std::uint32_t ahead_max = 0, ahead_top = 0;
    cudaStream_t ahead_stream = nullptr;
    std::vector<Ahead> ahead;
    std::vector<unsigned char> ahead_pick;
    // The guess the served narrow call found at its start besides its own layer's (the executor
    // joins visit v + 1's guess before visit v's split, and visit v + 2's starts after v is ready,
    // so it is the next request's layer) and its rows; -1: none. ahead_expect: the layer the last
    // read-ahead was for, checked against the next request.
    std::int64_t ahead_target = -1, ahead_expect = -1;
    std::int32_t ahead_rows   = 0;
    // NINFER_INK_PREFETCH_CHECK=1: every served call checks every table entry against the slots'
    // records; NINFER_INK_PREFETCH_SYNC=1: each read-ahead is waited for at once.
    bool ahead_check = false, ahead_sync = false;
    ExpertAheadStats ahead_counts;

    void ahead_add(const ExpertAheadStats& d) {
        std::lock_guard lock(mutex);
        auto& c = ahead_counts;
        c.issued += d.issued;
        c.used += d.used;
        c.waited += d.waited;
        c.wasted += d.wasted;
        c.no_room += d.no_room;
        c.to_lent += d.to_lent;
        c.mismatch += d.mismatch;
        c.ambiguous += d.ambiguous;
        c.narrow += d.narrow;
        c.serve_ns += d.serve_ns;
        c.wait_ns += d.wait_ns;
    }

    // A read-ahead copy has landed (or the copies stream waits for it): its table entries move to
    // its slot, unless the slot holds something else by now.
    void ahead_finish(const Ahead& r) {
        auto& layer = layers[r.layer];
        if (!r.lent) {
            auto& slot = slots[std::size_t(r.slot)];
            if (slot.layer == std::int32_t(r.layer) && slot.expert == r.expert && slot.pending) {
                slot.pending = false;
                point_slot(r.layer, r.expert, r.slot);
                tier->set(r.layer, r.expert, ExpertTier::State::Held);
            }
            return;
        }
        auto& slot = lent_slots[std::size_t(r.slot)];
        if (!slot.ahead || slot.layer != std::int32_t(r.layer) || slot.expert != r.expert) { return; }
        slot.ahead = false;
        if (layer.slot_of[std::size_t(r.expert)] >= 0 || layer.lent_of[std::size_t(r.expert)] >= 0) {
            // A round gave it a cache slot meanwhile: the lent slot is free again.
            slot.layer = slot.expert = -1;
            return;
        }
        point_lent(r.layer, r.expert, r.slot);
    }

    // Every read-ahead copy has landed (the stream is waited for): every entry moves.
    void ahead_flush(ExpertAheadStats& d) {
        if (ahead.empty()) { return; }
        CUDA_CHECK(ninfer::wait_for_stream(ahead_stream));
        for (const auto& r : ahead) {
            if (!r.judged) { ++d.wasted; }
            ahead_finish(r);
            spare_events.push_back(r.landed);
        }
        ahead.clear();
    }

    // A served call of layer li, after land(): read-ahead copies that landed move their entries;
    // one still in flight whose expert this call misses orders the call's copies (and so its
    // tables and its ready signal) after it. The guessed layer's call judges its copies.
    void ahead_settle(std::size_t li, const std::uint32_t* needed, bool wide, ExpertAheadStats& d) {
        if (wide) {
            ahead_flush(d);
            return;
        }
        for (auto it = ahead.begin(); it != ahead.end();) {
            auto& r = *it;
            const bool here = r.layer == li;
            const bool routed = here && call_routed[std::size_t(r.expert)] != 0;
            if (here && !r.judged) {
                r.judged = true;
                ++(routed ? d.used : d.wasted);
            }
            const cudaError_t status = cudaEventQuery(r.landed);
            if (status == cudaErrorNotReady) {
                if (!(routed && (needed[r.expert] & kMiss))) {
                    ++it;
                    continue;
                }
                CUDA_CHECK(cudaStreamWaitEvent(copies, r.landed, 0));
                ++d.waited;
            } else {
                CUDA_CHECK(status);
            }
            ahead_finish(r);
            spare_events.push_back(r.landed);
            it = ahead.erase(it);
        }
    }

    // A miss takes lent slot l while a read-ahead copy into it is in flight: the call's copies
    // follow that copy, and its expert is not read ahead after all.
    void ahead_cancel_lent(std::int32_t l) {
        ExpertAheadStats d;
        for (auto it = ahead.begin(); it != ahead.end(); ++it) {
            if (!it->lent || it->slot != l) { continue; }
            CUDA_CHECK(cudaStreamWaitEvent(copies, it->landed, 0));
            spare_events.push_back(it->landed);
            if (!it->judged) { ++d.wasted; }
            ahead.erase(it);
            break;
        }
        // Its expert's entries never pointed here: admit_lent must not point them nowhere.
        auto& slot  = lent_slots[std::size_t(l)];
        slot.ahead  = false;
        slot.layer  = slot.expert = -1;
        ahead_add(d);
    }

    // The lent slot a read-ahead copy for layer li takes: as next_lent() orders them, never one
    // in flight, one of layer li (its call may read it) or one the ready call routes.
    [[nodiscard]] std::int32_t ahead_lent(std::size_t li) const {
        std::int32_t best = -1;
        std::tuple<int, float, std::uint64_t> best_key{};
        for (std::size_t k = 0; k < bank_lent.size(); ++k) {
            if (!bank_lent[k]) { continue; }
            for (std::uint32_t i = 0; i < per_bank; ++i) {
                const auto l     = std::int32_t(k * per_bank + i);
                const auto& slot = lent_slots[std::size_t(l)];
                if (slot.ahead || slot.layer == std::int32_t(li)) { continue; }
                if (slot.layer >= 0 && slot.layer == std::int32_t(call_layer) &&
                    call_routed[std::size_t(slot.expert)]) {
                    continue;
                }
                const auto key =
                    slot.layer < 0
                        ? std::tuple{0, 0.0F, std::uint64_t(0)}
                        : std::tuple{options.loop_belady &&
                                             tier->guarded(std::size_t(slot.layer), slot.expert)
                                         ? 2 : 1,
                                     tier->count(std::size_t(slot.layer), slot.expert),
                                     slot.used};
                if (best < 0 || key < best_key) {
                    best     = l;
                    best_key = key;
                }
            }
        }
        return best;
    }

    // One guessed expert's read-ahead copy into a free or tier-admitted cache slot, else a lent
    // slot. The victim is never of layer li nor routed by the ready call: until the next call's
    // tables are uploaded only those two layers' entries are read. false: no slot.
    bool ahead_place(std::size_t li, std::int32_t e, ExpertDeliveryStats& delta,
                     ExpertAheadStats& d) {
        const auto& expert = layers[li].source.experts[std::size_t(e)];
        std::int32_t s = -1;
        auto& free = pool_of(li).free;
        if (!free.empty()) {
            s = free.back();
            free.pop_back();
        } else {
            const auto victim = tier->admission_victim(li, e, 1.0F, call_routed,
                                                       std::int64_t(call_layer), std::int64_t(li));
            if (victim.expert >= 0) {
                const auto v = layers[victim.layer].slot_of[std::size_t(victim.expert)];
                if (v >= 0 && !slots[std::size_t(v)].pending) {
                    evict(v, delta);
                    s = v;
                }
            }
        }
        Ahead r;
        r.layer  = li;
        r.expert = e;
        if (s >= 0) {
            occupy(s, li, e, true);
            for (std::size_t b = 0; b < expert.size(); ++b) {
                const auto& blob  = expert[b];
                std::byte* target = slot_blob(s, b);
                CUDA_CHECK(cudaMemcpyAsync(target, blob.pinned, blob.bytes, cudaMemcpyHostToDevice,
                                           ahead_stream));
                auto& written = slots[std::size_t(s)].written[b];
                if (written > blob.bytes) {
                    CUDA_CHECK(cudaMemsetAsync(target + blob.bytes, 0, written - blob.bytes,
                                               ahead_stream));
                }
                written = blob.bytes;
                delta.copied_bytes += blob.bytes;
            }
            ++delta.admissions;
            r.slot = s;
        } else {
            const std::int32_t l = any_lent() ? ahead_lent(li) : -1;
            if (l < 0) { return false; }
            auto& slot = lent_slots[std::size_t(l)];
            if (slot.layer >= 0 &&
                layers[std::size_t(slot.layer)].lent_of[std::size_t(slot.expert)] == l) {
                point_nowhere(std::size_t(slot.layer), slot.expert);
                ++delta.lent_evictions;
            }
            std::byte* base = lent_base(l);
            for (std::size_t b = 0; b < expert.size(); ++b) {
                const auto& blob = expert[b];
                CUDA_CHECK(cudaMemcpyAsync(base + slot_offset[b], blob.pinned, blob.bytes,
                                           cudaMemcpyHostToDevice, ahead_stream));
                auto& written = slot.written[b];
                if (written > blob.bytes) {
                    CUDA_CHECK(cudaMemsetAsync(base + slot_offset[b] + blob.bytes, 0,
                                               written - blob.bytes, ahead_stream));
                }
                written = blob.bytes;
                delta.copied_bytes += blob.bytes;
            }
            slot.layer  = std::int32_t(li);
            slot.expert = e;
            slot.ahead  = true;
            slot.used   = stamp;
            ++delta.lent_admissions;
            ++d.to_lent;
            r.slot = l;
            r.lent = true;
        }
        r.landed = take_event();
        CUDA_CHECK(cudaEventRecord(r.landed, ahead_stream));
        ahead.push_back(r);
        ++d.issued;
        return true;
    }

    // Once a narrow call is ready: the guess of the next request's layer (ahead_target) and up to
    // ahead_max of its guessed experts that no slot, lent slot or bank holds, best first.
    void ahead_issue() {
        if (ahead_target < 0) { return; }
        const std::size_t li     = std::size_t(ahead_target);
        const std::int32_t rows  = ahead_rows;
        ahead_target             = -1;
        if (li == call_layer || li >= layers.size()) { return; }
        ahead_expect             = std::int64_t(li);
        const std::size_t stride = 1 + std::size_t(guess_rows) * guess_width;
        const auto* area = static_cast<const volatile std::int32_t*>(guess_host->data());
        auto& layer       = layers[li];
        const int experts = int(layer.source.experts.size());
        const int held    = bank_holding(li);
        ahead_pick.assign(std::size_t(experts), 0);
        ExpertDeliveryStats delta;
        ExpertAheadStats d;
        std::uint32_t placed = 0;
        const std::uint32_t top = std::min(ahead_top, guess_width);
        for (std::uint32_t j = 0; j < top && placed < ahead_max; ++j) {
            for (std::int32_t r = 0; r < rows && placed < ahead_max; ++r) {
                const std::int32_t e = area[li * stride + 1 + std::size_t(r) * guess_width + j];
                if (e < 0 || e >= experts || ahead_pick[std::size_t(e)]) { continue; }
                ahead_pick[std::size_t(e)] = 1;
                if (layer.slot_of[std::size_t(e)] >= 0 || layer.lent_of[std::size_t(e)] >= 0 ||
                    layer.bank_of[std::size_t(e)] >= 0 ||
                    (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)])) {
                    continue;
                }
                const bool pending = std::any_of(ahead.begin(), ahead.end(), [&](const Ahead& a) {
                    return a.layer == li && a.expert == e;
                });
                if (pending) { continue; }
                const auto& blobs = layer.source.experts[std::size_t(e)];
                if (!std::all_of(blobs.begin(), blobs.end(),
                                 [](const ExpertBlob& b) { return b.pinned != nullptr; })) {
                    continue;
                }
                if (ahead_place(li, e, delta, d)) {
                    ++placed;
                } else {
                    ++d.no_room;
                }
            }
        }
        if (ahead_sync && placed) { CUDA_CHECK(ninfer::wait_for_stream(ahead_stream)); }
        add(delta);
        ahead_add(d);
    }

    // NINFER_INK_PREFETCH_CHECK=1: every table entry against the slots' records (the mirror the
    // served call uploads): null, its cache slot (held, not pending), its lent slot (not being
    // filled) or a bank. Throws on the first violations, after printing them.
    void ahead_verify(std::size_t li) {
        int bad = 0;
        const auto low  = reinterpret_cast<std::uintptr_t>(cache.p);
        const auto high = low + cache_span;
        for (std::size_t l = 0; l < layers.size() && bad < 8; ++l) {
            const auto& layer = layers[l];
            const auto n      = layer.source.experts.size();
            const auto** mirror = static_cast<const void**>(layer.mirror->data());
            const auto alias = layer.source.aliases[0];
            for (std::size_t e = 0; e < n && bad < 8; ++e) {
                const auto at = reinterpret_cast<std::uintptr_t>(mirror[e]);
                if (at == 0) { continue; }
                const char* why = nullptr;
                if (at >= low && at < high) {
                    const auto slot = layer.slot_of[e];
                    if (slot < 0) {
                        why = "cache entry without a slot";
                    } else if (slots[std::size_t(slot)].pending) {
                        why = "cache entry of a pending slot";
                    } else if (slots[std::size_t(slot)].layer != std::int32_t(l) ||
                               slots[std::size_t(slot)].expert != std::int32_t(e)) {
                        why = "cache slot holds another expert";
                    } else if (at != reinterpret_cast<std::uintptr_t>(
                                         slot_blob(slot, alias.blob) + alias.offset)) {
                        why = "cache entry off its slot";
                    } else if (std::uint32_t(slot - pools[slots[std::size_t(slot)].pool].first) >=
                               pools[slots[std::size_t(slot)].pool].count) {
                        why = "cache slot past the backed ones";
                    }
                } else if (layer.lent_of[e] >= 0) {
                    const auto& ls = lent_slots[std::size_t(layer.lent_of[e])];
                    if (ls.ahead) {
                        why = "lent entry of a slot being filled";
                    } else if (ls.layer != std::int32_t(l) || ls.expert != std::int32_t(e)) {
                        why = "lent slot holds another expert";
                    } else if (!bank_lent[std::size_t(layer.lent_of[e]) / per_bank]) {
                        why = "lent slot of a bank not lent";
                    }
                } else if (layer.bank_of[e] < 0) {
                    why = "entry outside cache, lent slots and banks";
                }
                if (why) {
                    std::fprintf(stderr, "hp-prefetch CHECK (call of layer %zu): layer %zu expert %zu: %s\n",
                                 li, l, e, why);
                    ++bad;
                }
            }
        }
        if (bad) { throw std::logic_error("expert delivery: NINFER_INK_PREFETCH_CHECK failed"); }
    }

    // A narrow call of layer li, before its misses are placed: the guess its layer holds against
    // the call's routes (needed) and the residency its split saw.
    void count_guess(std::size_t li, const std::uint32_t* needed) {
        const std::size_t stride = 1 + std::size_t(guess_rows) * guess_width;
        auto* slot = static_cast<volatile std::int32_t*>(guess_host->data()) + li * stride;
        const std::int32_t rows = slot[0];
        if (rows <= 0 || std::uint32_t(rows) > guess_rows) { return; }
        slot[0] = 0;
        const auto& layer = layers[li];
        const int experts = int(layer.source.experts.size());
        ExpertGuessStats d;
        d.calls = 1;
        for (int e = 0; e < experts; ++e) {
            if (!call_routed[std::size_t(e)]) { continue; }
            ++d.routes;
            if (needed[e] & kMiss) { ++d.misses; }
        }
        const std::uint32_t widths[2]{std::min<std::uint32_t>(8, guess_width), guess_width};
        for (int m = 0; m < 2; ++m) {
            guess_seen.assign(std::size_t(experts), 0);
            for (std::int32_t r = 0; r < rows; ++r) {
                for (std::uint32_t j = 0; j < widths[m]; ++j) {
                    const std::int32_t e = slot[1 + std::size_t(r) * guess_width + j];
                    if (e < 0 || e >= experts || guess_seen[std::size_t(e)]) { continue; }
                    guess_seen[std::size_t(e)] = 1;
                    const bool routed = call_routed[std::size_t(e)] != 0;
                    const auto s      = layer.slot_of[std::size_t(e)];
                    const bool resident =
                        routed ? (needed[e] & kHit) != 0
                               : (s >= 0 && !slots[std::size_t(s)].pending) ||
                                     layer.lent_of[std::size_t(e)] >= 0;
                    if (routed) {
                        ++d.routes_in[m];
                        if (needed[e] & kMiss) { ++d.misses_in[m]; }
                    }
                    if (!resident) {
                        ++d.open[m];
                        if (routed) { ++d.open_routed[m]; }
                    }
                }
            }
        }
        std::lock_guard lock(mutex);
        guess_counts.calls += d.calls;
        guess_counts.routes += d.routes;
        guess_counts.misses += d.misses;
        for (int m = 0; m < 2; ++m) {
            guess_counts.routes_in[m] += d.routes_in[m];
            guess_counts.misses_in[m] += d.misses_in[m];
            guess_counts.open[m] += d.open[m];
            guess_counts.open_routed[m] += d.open_routed[m];
        }
    }
    // hp-tune (miss-skip): the environment's limits (skip.weights stays null) and, over narrow
    // calls, the experts dropped, the misses before dropping and the routes before dropping;
    // a narrow call to layer 0 starts a pass, reported every 256.
    SplitSkip skip_limits;
    bool skip_on = false;
    std::uint64_t skip_passes = 0, skip_dropped = 0, skip_missing = 0, skip_routes = 0;
    std::vector<Pool> pools;
    // hp-tune (N6): per pool, its layers' profile routes, most first (empty: slots by layer).
    std::vector<std::vector<double>> pool_routes;
    // What backs a pool's slots: an elastic cache maps whole granules for each pool; one region
    // holds every pool's slots and rounds once.
    std::uint64_t unit         = 1;
    std::uint64_t expert_count = 0;
    Storage cache, cached_ids, missing_ids;
    // The call's expert lists, cached then missing (each its count, then the experts), and last
    // the handshake's sequence word.
    Storage lists;
    std::int32_t* cached_experts  = nullptr;
    std::int32_t* missing_experts = nullptr;
    // The addresses a table entry lies in when it points into the cache: an elastic cache's whole
    // reservation, so graphs captured before it grew still find its later slots.
    std::uint64_t cache_span = 0;
    std::vector<Storage> banks;
    std::vector<Bank> bank_state;
    int last_bank = 0;
    // Lent banks (ExpertDeliveryOptions::lend_banks): their slots, the slots one bank holds (0:
    // banks are never lent), which banks are lent, the device copy of their address ranges the
    // split reads and its pinned source, the narrow passes since the last wide call, and the
    // served calls' count.
    std::vector<LentSlot> lent_slots;
    std::uint32_t per_bank = 0;
    std::vector<unsigned char> bank_lent;
    Storage lent_ranges;
    std::unique_ptr<PinnedHostBuffer> host_ranges;
    bool ranges_dirty         = false;
    std::uint32_t quiet_passes = 0;
    std::uint64_t stamp        = 0;
    std::unique_ptr<PinnedHostBuffer> needed_host, staging;
    void* needed_mapped = nullptr;
    std::vector<Slot> slots; // every pool's, backed or not
    std::vector<Deferred> deferred;
    // The tier over the pools, the last narrow call's layer (a lower one starts a pass), and a
    // round's candidates whose copies post work has not issued yet.
    std::unique_ptr<ExpertTier> tier;
    std::int64_t last_narrow = -1;
    std::vector<Deferred> queued;
    cudaStream_t rounds = nullptr;
    std::vector<cudaEvent_t> spare_events;
    std::unique_ptr<PinnedHostBuffer> round_staging; // disk experts: a call's round reads
    cudaEvent_t round_staged = nullptr;               // the staging's last copies
    // A CPU share: the pool, the split, the call's inputs and ids on the host, the CPU's
    // products and flag, the corrected missing ids, and disk experts' bytes read for the CPU.
    std::unique_ptr<CpuSpinPool> pool;
    ExpertSplit split;
    std::unique_ptr<PinnedHostBuffer> host_inputs, host_ids, host_products, host_flag, host_missing;
    void* host_inputs_mapped = nullptr;
    void* host_ids_mapped    = nullptr;
    Storage products, cpu_flag;
    std::vector<std::byte> cpu_bytes;
    cudaEvent_t copies_begun = nullptr, copies_done = nullptr; // a call's copies, timed
    // The CPU workers spin (spinning) until cpu_until, 500 ms after the last call whose misses
    // they computed; a call with none (every expert cached, a wide call, no CPU share) keeps
    // them asleep, and a call that has some wakes them.
    bool spinning = false;
    std::chrono::steady_clock::time_point cpu_until{};
    std::optional<CudaHostHandshake> handshake;
    cudaStream_t copies = nullptr;
    cudaEvent_t copied[2]{};
    std::uint64_t half_bytes = 0;
    int half = 0;
    std::uint64_t staging_cursor = 0;
    std::vector<QueuedFileRead> batch_reads;
    struct StagedCopy { std::byte* target; const std::uint8_t* source; std::size_t bytes; };
    std::vector<StagedCopy> batch_copies;
    std::vector<std::ifstream> files; // the artifact's, then a drive mirror's
    std::unique_ptr<FileReadQueue> read_queue;
    std::size_t artifact_files = 0; // with a drive mirror, the artifact's file count
    std::vector<QueuedFileRead> striped_reads;
    // Direct reads (core/expert_direct.h): the window holds the two staging halves, then the round
    // area of round_window bytes; a batch's reads go to direct_reads instead of batch_reads.
    std::unique_ptr<DirectExpertWindow> direct;
    std::vector<DirectRead> direct_reads;
    std::uint64_t round_window = 0;
    std::atomic<bool> stopping{false};
    std::atomic<std::uint32_t> finished{0}; // the last request whose service, post work included, ended
    std::thread service;
    mutable std::mutex mutex;
    std::exception_ptr failure;
    ExpertDeliveryStats counters;
    ExpertDeliveryMemory allocation;
    bool released = false;
    // The call being served and its post work: layer, bank, whether wide, and its routed experts.
    // Between a call's ready signal and the next request the GPU may read any layer's table (the
    // next split, then that call's cached pairs), so post work writes only free slots. Cached
    // experts are replaced only while a request is served, never one it routes.
    std::int64_t previous_layer = -1;
    bool previous_wide          = false;
    std::size_t call_layer      = 0;
    int call_bank               = 0;
    bool call_wide              = false;
    std::vector<unsigned char> call_routed, call_cpu;
    std::vector<std::uint32_t> call_count;
    std::vector<Piece> pieces;
    std::vector<std::pair<std::byte*, std::uint64_t>> zeroes;

    Impl(const RankContext& r, std::vector<std::filesystem::path> paths,
         std::vector<ExpertDeliveryLayer> specs, ExpertDeliveryOptions o)
        : rank(r), options(o) {
        DeviceBinding bind(rank.device);
        // hp-tune (miss-skip): off unless a rank or a share is set.
        if (const char* v = std::getenv("NINFER_MISS_SKIP_RANK"); v && *v) {
            const long n = std::strtol(v, nullptr, 10);
            if (n >= 1) {
                skip_limits.rank = int(std::min(n, 1L << 30));
                skip_on          = true;
            }
        }
        if (const char* v = std::getenv("NINFER_MISS_SKIP_W"); v && *v) {
            const float w = std::strtof(v, nullptr);
            if (w > 0.0f && w < 1.0f) {
                skip_limits.share = w;
                skip_on           = true;
            }
        }
        if (const char* v = std::getenv("NINFER_MISS_SKIP_RENORM"); v && *v && *v != '0') {
            skip_limits.renorm = 1;
        }
        if (skip_on) {
            std::fprintf(stderr, "hp-tune: miss-skip rank=%d w=%g renorm=%d\n",
                         skip_limits.rank >= (1 << 30) ? -1 : skip_limits.rank,
                         double(skip_limits.share), skip_limits.renorm);
        }
        if (specs.empty() || options.pair_capacity == 0) {
            throw std::invalid_argument("expert delivery: layers and pair capacity are required");
        }
        if (options.banks < 1 || options.banks > 2) {
            throw std::invalid_argument("expert delivery: one or two layer banks");
        }
        bool disk = false;
        std::vector<std::uint64_t> largest, largest_tail;
        // Per layer and blob: the room a slot needs (its largest blob and tail, aligned).
        std::vector<std::vector<std::uint64_t>> rooms;
        for (const auto& layer : specs) {
            expert_count += layer.experts.size();
            if (layer.experts.empty() || layer.aliases.empty()) {
                throw std::invalid_argument("expert delivery: empty expert or alias table");
            }
            if (layer.experts.size() > 8192) {
                throw std::invalid_argument("expert delivery: at most 8192 experts per layer");
            }
            max_experts = std::max(max_experts, std::uint32_t(layer.experts.size()));
            const std::size_t blobs = layer.experts.front().size();
            std::vector<std::uint64_t> layer_largest(blobs, 0), layer_tail(blobs, 0);
            for (const auto& expert : layer.experts) {
                if (expert.empty() || expert.size() != blobs) {
                    throw std::invalid_argument("expert delivery: every expert of a layer needs its blobs");
                }
                largest.resize(std::max(largest.size(), expert.size()), 0);
                largest_tail.resize(largest.size(), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    const auto& blob = expert[b];
                    if (!blob.bytes || (blob.pinned != nullptr) == !blob.extents.empty()) {
                        throw std::invalid_argument("expert delivery: choose pinned bytes or extents");
                    }
                    std::uint64_t extent_bytes = 0;
                    for (const auto& extent : blob.extents) {
                        if (extent.file >= paths.size() || !extent.bytes ||
                            extent.bytes > blob.bytes - std::min(blob.bytes, extent_bytes)) {
                            throw std::invalid_argument("expert delivery: invalid file extent");
                        }
                        extent_bytes += extent.bytes;
                    }
                    if (!blob.pinned && extent_bytes != blob.bytes) {
                        throw std::invalid_argument("expert delivery: extents do not cover blob");
                    }
                    if (blob.zero_tail > std::numeric_limits<std::uint64_t>::max() - blob.bytes) {
                        throw std::invalid_argument("expert delivery: blob tail overflow");
                    }
                    largest[b]      = std::max(largest[b], blob.bytes);
                    largest_tail[b] = std::max(largest_tail[b], blob.zero_tail);
                    layer_largest[b] = std::max(layer_largest[b], blob.bytes);
                    layer_tail[b]    = std::max(layer_tail[b], blob.zero_tail);
                    disk |= !blob.extents.empty();
                }
                for (const auto& alias : layer.aliases) {
                    if (alias.blob >= expert.size() || alias.offset >= expert[alias.blob].bytes) {
                        throw std::invalid_argument("expert delivery: alias outside blob");
                    }
                }
            }
            auto& room = rooms.emplace_back(blobs);
            for (std::size_t b = 0; b < blobs; ++b) {
                room[b] = aligned(checked_add(layer_largest[b], layer_tail[b]));
            }
        }
        for (std::size_t b = 0; b < largest.size(); ++b) {
            slot_offset.push_back(slot_bytes);
            slot_size.push_back(aligned(checked_add(largest[b], largest_tail[b])));
            slot_bytes = checked_add(slot_bytes, slot_size.back());
        }
        tail = largest_tail;
        bank_size.assign(largest.size(), 0);
        if (options.cpu.products != nullptr &&
            (options.cpu.product_bytes == 0 || !(options.cpu.share >= 0 && options.cpu.share <= 1))) {
            throw std::invalid_argument("expert delivery: a CPU share needs its products and a share in [0, 1]");
        }
        // The largest expert's bytes, and its file reads' direct covers.
        std::uint64_t largest_expert = 0, largest_cover = 0;
        for (std::size_t index = 0; index < specs.size(); ++index) {
            auto& spec = specs[index];
            Layer layer;
            for (const auto& expert : spec.experts) {
                std::uint64_t bytes = 0, cover = 0;
                for (const auto& blob : expert) {
                    bytes += blob.bytes;
                    for (const auto& extent : blob.extents) {
                        cover += direct_cover(extent.offset, extent.bytes);
                    }
                }
                largest_expert = std::max(largest_expert, bytes);
                largest_cover  = std::max(largest_cover, cover);
            }
            layer.total.assign(spec.experts.size(), 0);
            layer.cpu = options.cpu.products != nullptr && options.cpu.products->supports(index);
            layer.pool = std::uint32_t(
                std::find_if(pools.begin(), pools.end(),
                             [&](const Pool& pool) { return pool.size == rooms[index]; }) -
                pools.begin());
            if (layer.pool == pools.size()) {
                auto& pool = pools.emplace_back();
                pool.size  = rooms[index];
                for (const auto size : pool.size) {
                    pool.offset.push_back(pool.slot_bytes);
                    pool.slot_bytes = checked_add(pool.slot_bytes, size);
                }
            }
            pools[layer.pool].capacity += std::uint32_t(spec.experts.size());
            ++pools[layer.pool].layers;
            const auto count = spec.experts.size();
            const auto blobs = spec.experts.front().size();
            for (std::size_t b = 0; b < blobs; ++b) {
                const auto first = spec.experts.front()[b].bytes;
                const bool equal = std::all_of(spec.experts.begin(), spec.experts.end(),
                    [&](const auto& expert) { return expert[b].bytes == first; });
                // Kernels read these planes as the resident bank: every expert at its index.
                const bool resident = equal && first % 4 == 0;
                layer.stride.push_back(resident ? first : aligned(largest[b]));
                bank_size[b] = std::max(bank_size[b],
                                        checked_add(checked_mul(count, layer.stride.back()), tail[b]));
            }
            layer.slot_of.assign(count, -1);
            layer.lent_of.assign(count, -1);
            layer.bank_of.assign(count, -1);
            const auto bytes = count * spec.aliases.size() * sizeof(void*);
            layer.mirror = std::make_unique<PinnedHostBuffer>(bytes);
            std::memset(layer.mirror->data(), 0, bytes);
            for (int u = 0; u < 2; ++u) {
                layer.upload[u] = std::make_unique<PinnedHostBuffer>(bytes);
                CUDA_CHECK(cudaEventCreateWithFlags(&layer.uploaded[u], cudaEventDisableTiming));
                allocation.pinned_bytes += bytes;
            }
            layer.source = std::move(spec);
            allocation.pinned_bytes += bytes;
            layers.push_back(std::move(layer));
        }
        for (const auto size : bank_size) {
            bank_offset.push_back(bank_bytes);
            bank_bytes = checked_add(bank_bytes, aligned(size));
        }
        cached_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        missing_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        lists =
            Storage((2 * std::size_t(1 + max_experts) + 1) * 4, rank.device, options.suspendable);
        lists.fill();
        cached_experts  = static_cast<std::int32_t*>(lists.p);
        missing_experts = cached_experts + 1 + max_experts;
        handshake.emplace(rank,
                          reinterpret_cast<std::uint32_t*>(missing_experts + 1 + max_experts));
        // Mapped: the split writes it.
        needed_host = std::make_unique<PinnedHostBuffer>(std::size_t(max_experts) * 4);
        std::memset(needed_host->data(), 0, needed_host->size());
        CUDA_CHECK(cudaHostGetDevicePointer(&needed_mapped, needed_host->data(), 0));
        allocation.metadata_bytes =
            cached_ids.backing_bytes() + missing_ids.backing_bytes() + lists.backing_bytes();
        allocation.pinned_bytes += needed_host->size() + handshake->pinned_bytes();
        for (auto& layer : layers) {
            const auto bytes = layer.mirror->size();
            layer.tables = Storage(bytes, rank.device, options.suspendable);
            layer.tables.fill();
            allocation.metadata_bytes += layer.tables.backing_bytes();
        }
        if (disk) {
            half_bytes = aligned(std::max<std::uint64_t>(options.staging_bytes / 2, 256));
            const auto mirror = expert_mirror_files(paths, options.disk.mirror);
            artifact_files    = mirror.empty() ? 0 : paths.size();
            std::vector<std::filesystem::path> all = paths;
            all.insert(all.end(), mirror.begin(), mirror.end());
            for (const auto& path : all) {
                files.emplace_back(path, std::ios::binary);
                if (!files.back()) {
                    throw std::runtime_error("expert delivery: cannot open " + path.string());
                }
            }
            read_queue = FileReadQueue::open(all, false, 16);
            // Direct reads fill halves of whole blocks and a round area that holds the largest
            // expert's covers; host staging is then unused.
            const std::uint64_t direct_half =
                checked_add(half_bytes, kDirectBlock - 1) / kDirectBlock * kDirectBlock;
            round_window = checked_add(std::max(kRoundFileBytes, largest_cover), kDirectBlock - 1) /
                           kDirectBlock * kDirectBlock;
            direct =
                open_direct_window(options.disk, rank.device, all,
                                   checked_add(2 * direct_half, round_window), "expert delivery");
            if (direct) {
                half_bytes = direct_half;
                allocation.metadata_bytes += direct->bytes();
            } else {
                staging = std::make_unique<PinnedHostBuffer>(2 * half_bytes);
                allocation.pinned_bytes += staging->size();
            }
        }
        const std::uint64_t granule = options.suspendable ? vmm::granularity(rank.device) : 1;
        if (!granule) {
            throw std::invalid_argument("expert delivery: suspend needs VMM allocation granularity");
        }
        const std::uint64_t cache_granule = options.elastic ? vmm::granularity(rank.device) : granule;
        if (!cache_granule) {
            throw std::invalid_argument("expert delivery: an elastic cache needs VMM allocation");
        }
        unit                = options.elastic ? cache_granule : 1;
        const auto physical = [&](std::uint64_t bytes) {
            return checked_add(bytes, granule - 1) / granule * granule;
        };
        const auto available = [&] {
            std::size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            return free > options.keep_free_bytes ? free - options.keep_free_bytes : 0;
        };
        // hp-tune (N6): the pools' routes per slot from the profile, most routed first.
        if (!options.slot_weights.empty()) {
            if (options.slot_weights.size() != layers.size()) {
                throw std::invalid_argument("expert delivery: slot weights describe other layers");
            }
            pool_routes.assign(pools.size(), {});
            for (std::size_t li = 0; li < layers.size(); ++li) {
                const auto& w = options.slot_weights[li];
                if (w.size() != layers[li].source.experts.size()) {
                    throw std::invalid_argument("expert delivery: slot weights of another bank width");
                }
                auto& routes = pool_routes[layers[li].pool];
                routes.insert(routes.end(), w.begin(), w.end());
            }
            for (auto& routes : pool_routes) { std::sort(routes.begin(), routes.end(), std::greater<>()); }
        }
        // Banks before the cache: every miss lands in one. A second bank is dropped before the
        // last one, and an explicit cache keeps its size by dropping the second bank first.
        std::uint32_t bank_count = options.banks;
        std::uint64_t explicit_cache = 0;
        if (options.cache_bytes) {
            const auto counts =
                fill_pools(*options.cache_bytes, std::vector<std::uint32_t>(pools.size(), 0));
            for (std::size_t p = 0; p < pools.size(); ++p) {
                explicit_cache = checked_add(explicit_cache, backing(pools[p], counts[p]));
            }
            explicit_cache = physical(explicit_cache);
        }
        while (bank_count > 1 && bank_count * physical(bank_bytes) + explicit_cache > available()) {
            --bank_count;
        }
        if (physical(bank_bytes) > available()) {
            throw std::invalid_argument(
                "expert delivery: a layer bank needs " + std::to_string(physical(bank_bytes)) +
                " device bytes, but only " + std::to_string(available()) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        banks.reserve(bank_count);
        for (std::uint32_t b = 0; b < bank_count; ++b) {
            banks.emplace_back(bank_bytes, rank.device, options.suspendable);
            banks.back().fill();
            bank_state.push_back({-1, std::vector<unsigned char>(max_experts, 0)});
            allocation.bank_bytes += banks.back().backing_bytes();
        }
        allocation.banks = bank_count;
        // A narrow call routes fewer than half a layer's experts, so one lent bank whose slots
        // number at least that many always has a slot for a miss that its call does not route.
        bank_lent.assign(bank_count, 0);
        if (options.lend_banks && 2 * (bank_bytes / slot_bytes) >= max_experts) {
            per_bank = std::uint32_t(bank_bytes / slot_bytes);
            lent_slots.resize(std::size_t(per_bank) * bank_count);
            for (auto& slot : lent_slots) { slot.written.assign(slot_offset.size(), 0); }
        }
        lent_ranges = Storage(4 * sizeof(std::uintptr_t), rank.device, options.suspendable);
        lent_ranges.fill();
        host_ranges = std::make_unique<PinnedHostBuffer>(4 * sizeof(std::uintptr_t));
        std::memset(host_ranges->data(), 0, host_ranges->size());
        allocation.metadata_bytes += lent_ranges.backing_bytes();
        allocation.pinned_bytes += host_ranges->size();
        const std::uint64_t remaining = available();
        // Automatic slots may use only whole physical granules inside the remaining budget.
        // An explicit request must preserve the same reserve, including VMM rounding.
        const auto budget = options.cache_bytes.value_or(remaining / cache_granule * cache_granule);
        const auto counts = fill_pools(budget, std::vector<std::uint32_t>(pools.size(), 0));
        std::uint64_t cache_bytes = 0, reserve = 0;
        std::int32_t first = 0;
        for (std::size_t p = 0; p < pools.size(); ++p) {
            auto& pool = pools[p];
            pool.first = first;
            first += std::int32_t(pool.capacity);
            pool.count = counts[p];
            if (options.elastic) {
                // Each pool's reservation starts at a granule, so its slots map and unmap alone.
                pool.span =
                    checked_add(checked_mul(pool.capacity, pool.slot_bytes), cache_granule - 1) /
                    cache_granule * cache_granule;
                pool.base = reserve;
                reserve   = checked_add(reserve, pool.span);
            } else {
                pool.span = checked_mul(pool.count, pool.slot_bytes);
                pool.base = cache_bytes;
            }
            cache_bytes = checked_add(cache_bytes, backing(pool, pool.count));
        }
        const auto backing_bytes = checked_add(cache_bytes, cache_granule - 1) / cache_granule * cache_granule;
        if (backing_bytes > remaining) {
            throw std::invalid_argument(
                "expert delivery: cache needs " + std::to_string(backing_bytes) +
                " device bytes, but only " + std::to_string(remaining) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        if (options.elastic) {
            cache.growable = GrowableRegion(reserve, {.device = rank.device});
            cache.p = cache.growable.data();
            for (const auto& pool : pools) {
                if (!cache.growable.map(pool.base, pool.count * pool.slot_bytes)) {
                    throw std::invalid_argument(
                        "expert delivery: the device cannot back the cache");
                }
            }
            cache.bytes = cache_bytes;
        }
        if (cache_bytes) {
            if (!options.elastic) { cache = Storage(cache_bytes, rank.device, options.suspendable); }
            cache.fill();
        }
        slots.resize(std::size_t(first));
        for (std::uint32_t p = 0; p < pools.size(); ++p) {
            auto& pool = pools[p];
            for (std::uint32_t i = 0; i < pool.capacity; ++i) {
                auto& slot = slots[std::size_t(pool.first) + i];
                slot.pool  = p;
                slot.written.assign(pool.offset.size(), 0);
            }
            reset_free(pool);
        }
        cache_span = options.elastic ? cache.growable.bytes() : cache.bytes;
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = cache_slots();
        if (!pool_routes.empty()) {
            std::string line;
            for (std::size_t p = 0; p < pools.size(); ++p) {
                line += " " + std::to_string(pools[p].count) + "/" + std::to_string(pools[p].layers) +
                        "L@" + std::to_string(pools[p].slot_bytes >> 10) + "K";
            }
            std::fprintf(stderr, "hp-tune: N6 slots by weight, pools (slots/layers@slot):%s\n",
                         line.c_str());
        }
        loop_visits.assign(layers.size(), 0);
        loop_prev.assign(layers.size(), std::vector<unsigned char>(max_experts, 0));
        if (options.loop_passes < 2 || options.loop_first >= options.loop_end ||
            options.loop_end > layers.size()) {
            options.loop_belady = options.loop_stats = false;
        }
        call_routed.assign(max_experts, 0);
        call_cpu.assign(max_experts, 0);
        call_count.assign(max_experts, 0);
        std::vector<std::uint32_t> widths, layer_pools;
        for (const auto& layer : layers) {
            widths.push_back(std::uint32_t(layer.source.experts.size()));
            layer_pools.push_back(layer.pool);
        }
        tier = std::make_unique<ExpertTier>(widths, layer_pools, options.tier);
        set_free();
        if (disk && !direct) {
            round_staging = std::make_unique<PinnedHostBuffer>(
                std::size_t(std::max(kRoundFileBytes, largest_expert)));
            allocation.pinned_bytes += round_staging->size();
        }
        if (options.cpu.products != nullptr) {
            const auto pinned = [&](std::size_t bytes) {
                auto buffer = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(bytes, 64));
                std::memset(buffer->data(), 0, buffer->size());
                allocation.pinned_bytes += buffer->size();
                return buffer;
            };
            host_inputs   = pinned(options.cpu.input_bytes);
            host_ids      = pinned(std::size_t(options.pair_capacity) * 4);
            // Mapped: the split writes them.
            CUDA_CHECK(cudaHostGetDevicePointer(&host_inputs_mapped, host_inputs->data(), 0));
            CUDA_CHECK(cudaHostGetDevicePointer(&host_ids_mapped, host_ids->data(), 0));
            host_missing  = pinned(std::size_t(options.pair_capacity + 1 + max_experts) * 4);
            host_products = pinned(options.cpu.product_bytes);
            host_flag     = pinned(4);
            products = Storage(options.cpu.product_bytes, rank.device, options.suspendable);
            cpu_flag = Storage(4, rank.device, options.suspendable);
            products.fill();
            cpu_flag.fill();
            allocation.metadata_bytes += products.backing_bytes() + cpu_flag.backing_bytes();
            pool = std::make_unique<CpuSpinPool>(
                options.cpu.threads != 0 ? options.cpu.threads : pool_cpu_count(), true);
        }
        CUDA_CHECK(cudaFuncSetAttribute(split_routes, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(max_experts * 4)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&rounds, cudaStreamNonBlocking));
        try {
            for (auto& event : copied) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
            CUDA_CHECK(cudaEventCreateWithFlags(&round_staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreate(&copies_begun));
            CUDA_CHECK(cudaEventCreate(&copies_done));
            service = std::thread([this] { run(); });
        } catch (...) {
            for (auto event : copied) { if (event) { cudaEventDestroy(event); } }
            for (auto event : {round_staged, copies_begun, copies_done}) {
                if (event) { cudaEventDestroy(event); }
            }
            cudaStreamDestroy(copies);
            cudaStreamDestroy(rounds);
            throw;
        }
    }

    ~Impl() {
        DeviceBinding bind(rank.device);
        // Keep the worker alive until the last graph's waits and consumers finish.
        ninfer::wait_for_stream(rank.stream);
        stopping.store(true, std::memory_order_release);
        if (service.joinable()) { service.join(); }
        ninfer::wait_for_stream(copies);
        ninfer::wait_for_stream(rounds);
        if (ahead_stream) {
            ninfer::wait_for_stream(ahead_stream);
            for (const auto& r : ahead) { spare_events.push_back(r.landed); }
            ahead.clear();
            cudaStreamDestroy(ahead_stream);
        }
        for (auto event : copied) { cudaEventDestroy(event); }
        for (auto& layer : layers) {
            for (auto event : layer.uploaded) { if (event) { cudaEventDestroy(event); } }
        }
        for (const auto& d : deferred) {
            if (d.landed && std::find(spare_events.begin(), spare_events.end(), d.landed) ==
                                spare_events.end()) {
                spare_events.push_back(d.landed);
            }
        }
        for (auto event : spare_events) { cudaEventDestroy(event); }
        for (auto event : {round_staged, copies_begun, copies_done}) { cudaEventDestroy(event); }
        cudaStreamDestroy(copies);
        cudaStreamDestroy(rounds);
    }

    void check() const {
        std::lock_guard lock(mutex);
        if (failure) { std::rethrow_exception(failure); }
    }

    template <typename Function> void each_storage(Function&& function) {
        for (Storage* storage : {&cache, &cached_ids, &missing_ids, &lists, &products, &cpu_flag,
                                 &lent_ranges}) {
            if (storage->bytes) { function(*storage); }
        }
        for (auto& bank : banks) { function(bank); }
        for (auto& layer : layers) { function(layer.tables); }
    }

    [[nodiscard]] std::byte* slot_base(std::int32_t slot) const {
        const auto& pool = pools[slots[std::size_t(slot)].pool];
        return static_cast<std::byte*>(cache.p) + pool.base +
               std::uint64_t(slot - pool.first) * pool.slot_bytes;
    }

    [[nodiscard]] std::byte* slot_blob(std::int32_t slot, std::size_t b) const {
        return slot_base(slot) + pools[slots[std::size_t(slot)].pool].offset[b];
    }

    [[nodiscard]] Pool& pool_of(std::size_t li) { return pools[layers[li].pool]; }

    // The bytes that back n slots of `pool` (`unit`).
    [[nodiscard]] std::uint64_t backing(const Pool& pool, std::uint64_t n) const {
        return checked_add(checked_mul(n, pool.slot_bytes), unit - 1) / unit * unit;
    }

    [[nodiscard]] std::uint32_t cache_slots() const {
        std::uint32_t n = 0;
        for (const auto& pool : pools) { n += pool.count; }
        return n;
    }

    // Every backed slot of `pool` free, popped lowest first.
    static void reset_free(Pool& pool) {
        pool.free.clear();
        for (std::int64_t i = std::int64_t(pool.count) - 1; i >= 0; --i) {
            pool.free.push_back(pool.first + std::int32_t(i));
        }
    }

    void set_free() {
        for (std::uint32_t p = 0; p < pools.size(); ++p) {
            tier->set_free(p, std::uint32_t(pools[p].free.size()));
        }
    }

    // Whether na slots of pool a give its layers fewer each than nb of pool b give b's (the lower
    // pool first among equals).
    [[nodiscard]] bool fewer(std::size_t a, std::uint32_t na, std::size_t b,
                             std::uint32_t nb) const {
        const auto x = std::uint64_t(na) * pools[b].layers, y = std::uint64_t(nb) * pools[a].layers;
        return x != y ? x < y : a < b;
    }

    // hp-tune (N6): the profile routes per byte of pool p's slot n (-1 past its experts).
    [[nodiscard]] double slot_value(std::size_t p, std::uint32_t n) const {
        const auto& routes = pool_routes[p];
        return n < routes.size() ? routes[n] / double(pools[p].slot_bytes) : -1.0;
    }
    // Whether pool a's next slot (after na) goes before pool b's (after nb).
    [[nodiscard]] bool before_in_fill(std::size_t a, std::uint32_t na, std::size_t b,
                                      std::uint32_t nb) const {
        if (!pool_routes.empty()) {
            const double va = slot_value(a, na), vb = slot_value(b, nb);
            if (va != vb) { return va > vb; }
        }
        return fewer(a, na, b, nb);
    }
    // Whether pool p's last slot (of np) leaves before pool w's (of nw).
    [[nodiscard]] bool before_in_trim(std::size_t p, std::uint32_t np, std::size_t w,
                                      std::uint32_t nw) const {
        if (!pool_routes.empty()) {
            const double vp = slot_value(p, np - 1), vw = slot_value(w, nw - 1);
            if (vp != vw) { return vp < vw; }
        }
        return fewer(w, nw, p, np);
    }

    // From `counts` (each pool's slots), one slot at a time to the pool whose layers hold the
    // fewest each (hp-tune N6: or whose next slot holds the most profile routes per byte), among
    // those with experts left whose next slot keeps the backing within
    // `bytes`: every layer about as many slots as any other, whatever its experts' size.
    [[nodiscard]] std::vector<std::uint32_t> fill_pools(std::uint64_t bytes,
                                                        std::vector<std::uint32_t> counts) const {
        std::uint64_t used = 0;
        for (std::size_t p = 0; p < pools.size(); ++p) { used += backing(pools[p], counts[p]); }
        for (;;) {
            std::size_t best = pools.size();
            for (std::size_t p = 0; p < pools.size(); ++p) {
                if (counts[p] >= pools[p].capacity) { continue; }
                const auto more = backing(pools[p], counts[p] + 1) - backing(pools[p], counts[p]);
                if (more > bytes - std::min(bytes, used)) { continue; }
                if (best == pools.size() || before_in_fill(p, counts[p], best, counts[best])) { best = p; }
            }
            if (best == pools.size()) { return counts; }
            used += backing(pools[best], counts[best] + 1) - backing(pools[best], counts[best]);
            ++counts[best];
        }
    }

    // From `counts`, one slot at a time from the pool whose layers hold the most each, until the
    // backing is within `bytes`.
    [[nodiscard]] std::vector<std::uint32_t> trim_pools(std::uint64_t bytes,
                                                        std::vector<std::uint32_t> counts) const {
        std::uint64_t used = 0;
        for (std::size_t p = 0; p < pools.size(); ++p) { used += backing(pools[p], counts[p]); }
        while (used > bytes) {
            std::size_t worst = pools.size();
            for (std::size_t p = 0; p < pools.size(); ++p) {
                if (counts[p] == 0) { continue; }
                if (worst == pools.size() || before_in_trim(p, counts[p], worst, counts[worst])) {
                    worst = p;
                }
            }
            if (worst == pools.size()) { break; }
            used -= backing(pools[worst], counts[worst]) - backing(pools[worst], counts[worst] - 1);
            --counts[worst];
        }
        return counts;
    }
    [[nodiscard]] std::byte* bank_blob(int bank, std::size_t li, std::int32_t e, std::size_t b) const {
        return static_cast<std::byte*>(banks[std::size_t(bank)].p) + bank_offset[b] +
               std::uint64_t(e) * layers[li].stride[b];
    }

    // The table entries of (li, e) point at base(b) for each alias's blob, or at nothing.
    template <typename Base> void point(std::size_t li, std::int32_t e, Base&& base) {
        auto& layer = layers[li];
        const auto n = layer.source.experts.size();
        auto** mirror = static_cast<const void**>(layer.mirror->data());
        for (std::size_t a = 0; a < layer.source.aliases.size(); ++a) {
            const auto alias = layer.source.aliases[a];
            std::byte* blob = base(alias.blob);
            mirror[a * n + std::size_t(e)] = blob ? blob + alias.offset : nullptr;
        }
        layer.dirty = true;
    }
    // Before (li, e)'s entries point elsewhere: the lent slot holding it is free again.
    void leave(std::size_t li, std::int32_t e) {
        auto& layer = layers[li];
        layer.bank_of[std::size_t(e)] = -1;
        const auto l = layer.lent_of[std::size_t(e)];
        if (l < 0) { return; }
        layer.lent_of[std::size_t(e)] = -1;
        lent_slots[std::size_t(l)].layer = lent_slots[std::size_t(l)].expert = -1;
    }
    void point_slot(std::size_t li, std::int32_t e, std::int32_t slot) {
        leave(li, e);
        point(li, e, [&](std::size_t b) { return slot_blob(slot, b); });
    }
    void point_bank(std::size_t li, std::int32_t e, int bank) {
        leave(li, e);
        point(li, e, [&](std::size_t b) { return bank_blob(bank, li, e, b); });
        layers[li].bank_of[std::size_t(e)] = std::int8_t(bank);
    }
    void point_nowhere(std::size_t li, std::int32_t e) {
        leave(li, e);
        point(li, e, [](std::size_t) { return static_cast<std::byte*>(nullptr); });
    }
    [[nodiscard]] std::byte* lent_base(std::int32_t l) const {
        const auto bank = std::uint32_t(l) / per_bank;
        return static_cast<std::byte*>(banks[bank].p) + std::uint64_t(std::uint32_t(l) % per_bank) * slot_bytes;
    }
    void point_lent(std::size_t li, std::int32_t e, std::int32_t l) {
        leave(li, e);
        std::byte* base = lent_base(l);
        point(li, e, [&](std::size_t b) { return base + slot_offset[b]; });
        layers[li].lent_of[std::size_t(e)] = l;
        auto& slot = lent_slots[std::size_t(l)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.used = stamp;
    }

    // Lends bank k: nothing points into its layout any more, and its zeroed bytes hold free slots.
    // The served call's split read every such entry as a miss, and earlier calls' consumers are done.
    void lend_bank(int k) {
        for (std::size_t li = 0; li < layers.size(); ++li) {
            for (std::size_t e = 0; e < layers[li].bank_of.size(); ++e) {
                if (layers[li].bank_of[e] == k) { point_nowhere(li, std::int32_t(e)); }
            }
        }
        bank_state[std::size_t(k)].layer = -1;
        std::fill(bank_state[std::size_t(k)].valid.begin(), bank_state[std::size_t(k)].valid.end(), 0);
        CUDA_CHECK(cudaMemsetAsync(banks[std::size_t(k)].p, 0, banks[std::size_t(k)].bytes, copies));
        for (std::uint32_t i = 0; i < per_bank; ++i) {
            auto& slot = lent_slots[std::size_t(k) * per_bank + i];
            slot.layer = slot.expert = -1;
            std::fill(slot.written.begin(), slot.written.end(), 0);
        }
        bank_lent[std::size_t(k)] = 1;
        auto* ranges = static_cast<std::uintptr_t*>(host_ranges->data());
        ranges[2 * k]     = reinterpret_cast<std::uintptr_t>(banks[std::size_t(k)].p);
        ranges[2 * k + 1] = ranges[2 * k] + std::uint64_t(per_bank) * slot_bytes;
        ranges_dirty = true;
        set_lent_count();
    }

    // A wide call takes bank k back: the experts of its slots leave (the call read their entries
    // as misses).
    void reclaim_bank(int k, ExpertDeliveryStats& delta) {
        for (std::uint32_t i = 0; i < per_bank; ++i) {
            const auto& slot = lent_slots[std::size_t(k) * per_bank + i];
            if (slot.layer < 0) { continue; }
            point_nowhere(std::size_t(slot.layer), slot.expert);
            ++delta.lent_evictions;
        }
        bank_lent[std::size_t(k)] = 0;
        auto* ranges = static_cast<std::uintptr_t*>(host_ranges->data());
        ranges[2 * k] = ranges[2 * k + 1] = 0;
        ranges_dirty = true;
        set_lent_count();
    }

    void set_lent_count() {
        std::uint32_t lent = 0;
        for (const auto flag : bank_lent) { lent += flag ? per_bank : 0; }
        std::lock_guard lock(mutex);
        allocation.lent_slots = lent;
    }

    [[nodiscard]] bool any_lent() const {
        return std::any_of(bank_lent.begin(), bank_lent.end(), [](unsigned char flag) { return flag != 0; });
    }

    // The lent slot a narrow call of layer li fills next, other than those it took (`taken`): a
    // free one, else the least routed expert (the tier's counts, then the oldest use) that the call
    // does not route, the lower slot first among equals. -1: none. A call's misses take them one
    // by one; nothing they change moves another slot's place in this order.
    [[nodiscard]] std::int32_t next_lent(std::size_t li,
                                         std::span<const std::int32_t> taken) const {
        std::int32_t best = -1;
        std::tuple<int, float, std::uint64_t> best_key{};
        for (std::size_t k = 0; k < bank_lent.size(); ++k) {
            if (!bank_lent[k]) { continue; }
            for (std::uint32_t i = 0; i < per_bank; ++i) {
                const auto l = std::int32_t(k * per_bank + i);
                const auto& slot = lent_slots[std::size_t(l)];
                if (slot.layer == std::int32_t(li) && call_routed[std::size_t(slot.expert)]) { continue; }
                if (std::find(taken.begin(), taken.end(), l) != taken.end()) { continue; }
                // hp-tune (N12): a guarded expert (due again this token) only after the others.
                // hp-tune (N11''): a slot a read-ahead copy fills, last of all.
                const auto key =
                    slot.ahead ? std::tuple{3, 0.0F, std::uint64_t(0)}
                    : slot.layer < 0
                        ? std::tuple{0, 0.0F, std::uint64_t(0)}
                        : std::tuple{options.loop_belady &&
                                             tier->guarded(std::size_t(slot.layer), slot.expert)
                                         ? 2 : 1,
                                     tier->count(std::size_t(slot.layer), slot.expert),
                                     slot.used};
                if (best < 0 || key < best_key) {
                    best     = l;
                    best_key = key;
                }
            }
        }
        return best;
    }

    // Copies a narrow call's missing expert e of layer li into lent slot l.
    void admit_lent(std::int32_t l, std::size_t li, std::int32_t e, ExpertDeliveryStats& delta) {
        if (lent_slots[std::size_t(l)].ahead) { ahead_cancel_lent(l); } // hp-tune (N11'')
        const auto& slot = lent_slots[std::size_t(l)];
        if (slot.layer >= 0 && options.loop_belady &&
            tier->guarded(std::size_t(slot.layer), slot.expert)) {
            ++loop_forced; // hp-tune (N12): every lent slot held a guarded expert
        }
        if (slot.layer >= 0) {
            point_nowhere(std::size_t(slot.layer), slot.expert);
            ++delta.lent_evictions;
        }
        auto& written = lent_slots[std::size_t(l)].written;
        std::byte* base = lent_base(l);
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto& blob = layers[li].source.experts[std::size_t(e)][b];
            add_piece(base + slot_offset[b], blob);
            if (written[b] > blob.bytes) {
                zeroes.push_back({base + slot_offset[b] + blob.bytes, written[b] - blob.bytes});
            }
            written[b] = blob.bytes;
        }
        point_lent(li, e, l);
        ++delta.lent_admissions;
    }

    // One copy per changed table, from a pinned copy no earlier upload still reads.
    void upload_tables() {
        for (auto& layer : layers) {
            if (!layer.dirty) { continue; }
            const int u = layer.next_upload;
            layer.next_upload = 1 - u;
            CUDA_CHECK(ninfer::wait_for_event(layer.uploaded[u]));
            std::memcpy(layer.upload[u]->data(), layer.mirror->data(), layer.mirror->size());
            CUDA_CHECK(cudaMemcpyAsync(layer.tables.p, layer.upload[u]->data(), layer.mirror->size(),
                                       cudaMemcpyHostToDevice, copies));
            CUDA_CHECK(cudaEventRecord(layer.uploaded[u], copies));
            layer.dirty = false;
        }
    }

    // Queue a host-to-device copy, extending the previous piece when both its source and its
    // target continue it.
    void add_piece(std::byte* target, const ExpertBlob& blob) {
        if (!pieces.empty()) {
            auto& last = pieces.back();
            const bool target_follows = last.target + last.bytes == target;
            if (target_follows && blob.pinned && last.pinned && blob.pinned_region != 0 &&
                blob.pinned_region == last.region &&
                last.pinned + last.bytes == static_cast<const std::byte*>(blob.pinned)) {
                last.bytes += blob.bytes;
                return;
            }
            if (target_follows && !blob.pinned && !last.pinned) {
                for (const auto& extent : blob.extents) {
                    auto& end = last.extents.back();
                    if (end.file == extent.file && end.offset + end.bytes == extent.offset) {
                        end.bytes += extent.bytes;
                    } else {
                        last.extents.push_back(extent);
                    }
                }
                last.bytes += blob.bytes;
                return;
            }
        }
        Piece piece;
        piece.target = target;
        piece.pinned = static_cast<const std::byte*>(blob.pinned);
        piece.region = blob.pinned_region;
        piece.extents = blob.extents;
        piece.bytes = blob.bytes;
        pieces.push_back(std::move(piece));
    }

    // Several reads share one bounded half. FileReadQueue can submit their extents together.
    // Its fallback reads the same ranges through the OS page cache.
    // With a drive mirror, a read whose offset lies in an odd stripe goes to the mirror's copy.
    void read_now(std::span<const QueuedFileRead> reads) {
        if (artifact_files != 0) {
            striped_reads.assign(reads.begin(), reads.end());
            for (auto& read : striped_reads) {
                if (expert_mirror_copy(read.offset) == 1) { read.file += artifact_files; }
            }
            reads = striped_reads;
        }
        if (read_queue) {
            read_queue->read(reads);
            return;
        }
        for (const auto& read : reads) {
            auto& file = files.at(read.file);
            file.clear();
            file.seekg(std::streamoff(read.offset));
            file.read(reinterpret_cast<char*>(read.destination.data()),
                      std::streamsize(read.destination.size()));
            if (!file) { throw std::runtime_error("expert delivery: incomplete file read"); }
        }
    }

    // The window file a direct read of `file` at `offset` goes to: with a drive mirror, the
    // mirror's copy for an odd stripe, as read_now() picks.
    [[nodiscard]] std::size_t window_file(std::size_t file, std::uint64_t offset) const noexcept {
        return artifact_files != 0 && expert_mirror_copy(offset) == 1 ? file + artifact_files
                                                                      : file;
    }

    [[nodiscard]] const std::uint8_t* in_window(std::uint64_t at) const noexcept {
        return reinterpret_cast<const std::uint8_t*>(direct->data() + at);
    }

    void flush_reads() {
        if (batch_reads.empty() && direct_reads.empty()) { return; }
        CUDA_CHECK(ninfer::wait_for_event(copied[half]));
        if (direct) {
            direct->read(direct_reads);
        } else {
            read_now(batch_reads);
        }
        for (const auto& copy : batch_copies) {
            CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                       direct ? cudaMemcpyDefault : cudaMemcpyHostToDevice,
                                       copies));
        }
        CUDA_CHECK(cudaEventRecord(copied[half], copies));
        batch_reads.clear();
        direct_reads.clear();
        batch_copies.clear();
        staging_cursor = 0;
        half = 1 - half;
    }

    // Direct reads: each extent's cover in the half being filled (whole blocks, so the cursor
    // stays on a block and a half's room always exceeds a lead), its copy out of the window.
    void stage_direct(const Piece& piece, ExpertDeliveryStats& delta) {
        std::uint64_t destination = 0;
        for (const auto& extent : piece.extents) {
            std::uint64_t within = 0;
            while (within < extent.bytes) {
                if (staging_cursor == half_bytes) { flush_reads(); }
                const std::uint64_t offset = extent.offset + within;
                const std::uint64_t at     = half * half_bytes + staging_cursor;
                const auto bytes = std::min(extent.bytes - within,
                                            half_bytes - staging_cursor - direct_lead(offset));
                direct_reads.push_back({window_file(extent.file, offset), offset, bytes, at});
                batch_copies.push_back({piece.target + destination,
                                        in_window(at + direct_lead(offset)), std::size_t(bytes)});
                staging_cursor += direct_cover(offset, bytes);
                destination += bytes;
                within += bytes;
                delta.read_bytes += bytes;
            }
        }
    }

    void stage(const Piece& piece, ExpertDeliveryStats& delta) {
        if (direct) {
            stage_direct(piece, delta);
            return;
        }
        std::uint64_t destination = 0;
        for (const auto& extent : piece.extents) {
            std::uint64_t within = 0;
            while (within < extent.bytes) {
                if (staging_cursor == half_bytes) { flush_reads(); }
                auto* buffer = static_cast<std::uint8_t*>(staging->data()) +
                               half * half_bytes + staging_cursor;
                const auto bytes = std::min(extent.bytes - within, half_bytes - staging_cursor);
                batch_reads.push_back({extent.file, extent.offset + within,
                                       {buffer, std::size_t(bytes)}});
                batch_copies.push_back({piece.target + destination, buffer, std::size_t(bytes)});
                staging_cursor += bytes;
                destination += bytes;
                within += bytes;
                delta.read_bytes += bytes;
            }
        }
    }

    void flush_pieces(ExpertDeliveryStats& delta) {
        for (const auto& piece : pieces) {
            if (piece.pinned) {
                CUDA_CHECK(cudaMemcpyAsync(piece.target, piece.pinned, piece.bytes,
                                           cudaMemcpyHostToDevice, copies));
            } else {
                stage(piece, delta);
            }
            delta.copied_bytes += piece.bytes;
        }
        flush_reads();
        for (const auto& [target, bytes] : zeroes) {
            CUDA_CHECK(cudaMemsetAsync(target, 0, bytes, copies));
        }
        pieces.clear();
        zeroes.clear();
    }

    // Bytes of blob b written into slot s: zero what a larger earlier blob left past them, so the
    // slot's tail stays zero.
    void slot_written(std::int32_t s, std::size_t b, std::uint64_t bytes) {
        auto& written = slots[std::size_t(s)].written[b];
        if (written > bytes) { zeroes.push_back({slot_blob(s, b) + bytes, written - bytes}); }
        written = bytes;
    }

    void evict(std::int32_t s, ExpertDeliveryStats& delta) {
        auto& slot = slots[std::size_t(s)];
        auto& layer = layers[std::size_t(slot.layer)];
        layer.slot_of[std::size_t(slot.expert)] = -1;
        --layer.cached;
        point_nowhere(std::size_t(slot.layer), slot.expert);
        tier->set(std::size_t(slot.layer), slot.expert, ExpertTier::State::Out);
        slot.layer = slot.expert = -1;
        slot.pending = false;
        ++delta.evictions;
    }

    // A free slot for the served call's missing expert e of layer li, routed `pairs` times, else
    // the slot of the coldest expert the call does not route when the tier admits e. -1: none.
    std::int32_t admit_slot(std::size_t li, std::int32_t e, std::uint32_t pairs,
                            ExpertDeliveryStats& delta) {
        auto& free = pool_of(li).free;
        if (!free.empty()) {
            const auto s = free.back();
            free.pop_back();
            return s;
        }
        const auto victim = tier->admission_victim(li, e, float(pairs), call_routed);
        if (victim.expert < 0) { return -1; }
        const auto s = layers[victim.layer].slot_of[std::size_t(victim.expert)];
        if (s < 0 || slots[std::size_t(s)].pending) { return -1; }
        evict(s, delta);
        return s;
    }

    void occupy(std::int32_t s, std::size_t li, std::int32_t e, bool pending) {
        auto& slot = slots[std::size_t(s)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.pending = pending;
        layers[li].slot_of[std::size_t(e)] = s;
        ++layers[li].cached;
        tier->set(li, e, pending ? ExpertTier::State::Pending : ExpertTier::State::Held);
    }

    cudaEvent_t take_event() {
        if (!spare_events.empty()) {
            const auto event = spare_events.back();
            spare_events.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }

    // Promotions of an earlier wide call (their copies precede the next table upload on
    // `copies`) and round candidates whose copies have landed: their entries move to their slots.
    // With `keep_hits`, an expert the served narrow call routes from a lent slot stays there until
    // a later call: the call's cached pairs may still be reading that slot, which a miss of the
    // call would otherwise take once its expert left.
    void land(bool keep_hits = false) {
        std::vector<cudaEvent_t> done;
        for (auto it = deferred.begin(); it != deferred.end();) {
            if (keep_hits && it->layer == call_layer && call_routed[std::size_t(it->expert)] &&
                layers[it->layer].lent_of[std::size_t(it->expert)] >= 0) {
                ++it;
                continue;
            }
            if (it->landed) {
                const cudaError_t status = cudaEventQuery(it->landed);
                if (status == cudaErrorNotReady) {
                    ++it;
                    continue;
                }
                CUDA_CHECK(status);
                if (std::find(done.begin(), done.end(), it->landed) == done.end()) {
                    done.push_back(it->landed);
                }
            }
            auto& slot = slots[std::size_t(it->slot)];
            if (slot.layer == std::int32_t(it->layer) && slot.expert == it->expert && slot.pending) {
                slot.pending = false;
                point_slot(it->layer, it->expert, it->slot);
                tier->set(it->layer, it->expert, ExpertTier::State::Held);
            }
            it = deferred.erase(it);
        }
        for (const auto event : done) {
            const bool used = std::any_of(deferred.begin(), deferred.end(),
                                          [&](const Deferred& d) { return d.landed == event; });
            if (!used) { spare_events.push_back(event); }
        }
    }

    // hp-tune (N11/N12): before a call's misses are placed. A wide call ends the token's guards;
    // a narrow one may start a token, and at a loop layer it moves the layer's guards to the
    // experts it routes when the token visits the layer again.
    void loop_begin(std::size_t li, bool wide) {
        loop_visit = -1;
        if (!options.loop_belady && !options.loop_stats) { return; }
        if (wide) {
            if (options.loop_belady) { tier->clear_guards(); }
            std::fill(loop_visits.begin(), loop_visits.end(), 0);
            return;
        }
        if (last_narrow < 0 ||
            (std::int64_t(li) < last_narrow && li < std::size_t(options.loop_first))) {
            if (options.loop_belady) { tier->clear_guards(); }
            std::fill(loop_visits.begin(), loop_visits.end(), 0);
            if (++loop_tokens % 256 == 0) { loop_report(); }
        }
        if (li < options.loop_first || li >= options.loop_end) { return; }
        loop_visit = std::int64_t(loop_visits[li]++);
        if (!options.loop_belady) { return; }
        const auto n = std::int32_t(layers[li].source.experts.size());
        const bool again = std::uint32_t(loop_visit) + 1 < options.loop_passes;
        for (std::int32_t e = 0; e < n; ++e) {
            tier->guard(li, e, again && call_routed[std::size_t(e)] != 0);
        }
    }

    // hp-tune (N11): after a loop layer's misses are placed: what its pass shares with the pass
    // before, and where those experts were.
    void loop_end_call(std::size_t li, const std::uint32_t* needed) {
        if (loop_visit < 0) { return; }
        auto& prev = loop_prev[li];
        const auto n = layers[li].source.experts.size();
        if (options.loop_stats && loop_visit >= 1) {
            auto& c = loop_counts[std::min<std::int64_t>(loop_visit - 1, 2)];
            ++c.visits;
            for (std::size_t e = 0; e < n; ++e) {
                c.previous += prev[e];
                if (!call_routed[e]) { continue; }
                ++c.routes;
                const bool hit = (needed[e] & kHit) != 0;
                if (prev[e]) {
                    ++c.repeat;
                    if (hit) {
                        ++(layers[li].lent_of[e] >= 0 ? c.repeat_lent : c.repeat_cache);
                    } else {
                        ++(call_cpu[e] ? c.repeat_cpu : c.repeat_copied);
                    }
                } else if (!hit) {
                    ++(call_cpu[e] ? c.fresh_cpu : c.fresh_copied);
                }
            }
        }
        for (std::size_t e = 0; e < n; ++e) { prev[e] = call_routed[e]; }
    }

    void loop_report() const {
        if (options.loop_stats) {
            for (std::uint32_t t = 0; t + 1 < std::min<std::uint32_t>(options.loop_passes, 4); ++t) {
                const auto& c = loop_counts[t];
                if (c.visits == 0) { continue; }
                const double r = double(std::max<std::uint64_t>(c.repeat, 1));
                std::fprintf(stderr,
                             "hp-tune: N11 loop pass %u->%u after %llu tokens: %.2f routes/visit, "
                             "%.1f%% routed by the pass before (%.2f/visit): of those %.1f%% "
                             "cache, %.1f%% lent, %.1f%% CPU, %.1f%% copied; new experts per "
                             "visit: %.2f CPU, %.2f copied\n",
                             t + 1, t + 2, (unsigned long long)loop_tokens,
                             double(c.routes) / double(c.visits),
                             100.0 * double(c.repeat) / double(std::max<std::uint64_t>(c.routes, 1)),
                             double(c.repeat) / double(c.visits), 100.0 * double(c.repeat_cache) / r,
                             100.0 * double(c.repeat_lent) / r, 100.0 * double(c.repeat_cpu) / r,
                             100.0 * double(c.repeat_copied) / r,
                             double(c.fresh_cpu) / double(c.visits),
                             double(c.fresh_copied) / double(c.visits));
            }
        }
        if (options.loop_belady) {
            std::fprintf(stderr,
                         "hp-tune: N12 after %llu tokens: %llu guarded admission victims passed "
                         "over, %llu guarded lent slots taken (no other)\n",
                         (unsigned long long)loop_tokens, (unsigned long long)tier->guard_skips(),
                         (unsigned long long)loop_forced);
        }
    }

    // A pass has ended: the tier's round. Victims leave now (never an expert the served call
    // routes); candidates hold their slots pending until post work has copied them in.
    void round(ExpertDeliveryStats& delta) {
        set_free();
        for (const auto& swap : tier->end_pass()) {
            auto& layer = layers[swap.layer];
            if (layer.slot_of[std::size_t(swap.in)] >= 0) { continue; }
            std::int32_t s = -1;
            if (swap.out < 0) {
                auto& free = pools[layer.pool].free;
                if (free.empty()) { continue; }
                s = free.back();
                free.pop_back();
            } else {
                s = layers[swap.out_layer].slot_of[std::size_t(swap.out)];
                if (s < 0 || slots[std::size_t(s)].pending ||
                    (swap.out_layer == call_layer && call_routed[std::size_t(swap.out)]) ||
                    (options.loop_belady && tier->guarded(swap.out_layer, swap.out))) {
                    continue; // hp-tune (N12): a guarded victim stays
                }
                evict(s, delta);
            }
            occupy(s, swap.layer, swap.in, true);
            queued.push_back({swap.layer, swap.in, s, nullptr});
            ++delta.admissions;
        }
    }

    // Post work of a narrow call: copies of the oldest queued candidates on the side stream, a few
    // experts and file bytes at a time, so the next call's service is not delayed.
    void issue_rounds(ExpertDeliveryStats& delta) {
        if (queued.empty()) { return; }
        const cudaEvent_t done = take_event();
        std::uint64_t staged = 0;
        std::size_t issued = 0;
        std::vector<QueuedFileRead> reads;
        std::vector<DirectRead> window_reads;
        std::vector<StagedCopy> staged_copies;
        auto* staging = round_staging ? static_cast<std::uint8_t*>(round_staging->data()) : nullptr;
        // Direct reads: the round area follows the window's two halves; `staged` counts covers.
        const std::uint64_t round_base = 2 * half_bytes;
        const std::uint64_t room = direct ? round_window : (staging ? round_staging->size() : 0);
        for (; issued < queued.size() && issued < kRoundExperts; ++issued) {
            const auto& d = queued[issued];
            const auto& expert = layers[d.layer].source.experts[std::size_t(d.expert)];
            std::uint64_t file_bytes = 0;
            for (const auto& blob : expert) {
                if (blob.pinned) { continue; }
                if (!direct) {
                    file_bytes += blob.bytes;
                    continue;
                }
                for (const auto& extent : blob.extents) {
                    file_bytes += direct_cover(extent.offset, extent.bytes);
                }
            }
            if (file_bytes != 0 && staged != 0 &&
                (staged + file_bytes > room || staged + file_bytes > kRoundFileBytes)) {
                break;
            }
            if (file_bytes != 0 && staged == 0) {
                CUDA_CHECK(ninfer::wait_for_event(round_staged));
            }
            for (std::size_t b = 0; b < expert.size(); ++b) {
                const auto& blob = expert[b];
                std::byte* target = slot_blob(d.slot, b);
                if (blob.pinned) {
                    CUDA_CHECK(cudaMemcpyAsync(target, blob.pinned, blob.bytes, cudaMemcpyHostToDevice,
                                               rounds));
                } else if (direct) {
                    std::uint64_t at = 0;
                    for (const auto& extent : blob.extents) {
                        const std::uint64_t window_at = round_base + staged;
                        window_reads.push_back({window_file(extent.file, extent.offset),
                                                extent.offset, extent.bytes, window_at});
                        staged_copies.push_back({target + at,
                                                 in_window(window_at + direct_lead(extent.offset)),
                                                 std::size_t(extent.bytes)});
                        staged += direct_cover(extent.offset, extent.bytes);
                        at += extent.bytes;
                    }
                    delta.read_bytes += blob.bytes;
                } else {
                    std::uint64_t at = 0;
                    for (const auto& extent : blob.extents) {
                        reads.push_back({extent.file, extent.offset,
                                         {staging + staged + at, std::size_t(extent.bytes)}});
                        at += extent.bytes;
                    }
                    staged_copies.push_back({target, staging + staged, std::size_t(blob.bytes)});
                    staged += blob.bytes;
                    delta.read_bytes += blob.bytes;
                }
                auto& written = slots[std::size_t(d.slot)].written[b];
                if (written > blob.bytes) {
                    CUDA_CHECK(cudaMemsetAsync(target + blob.bytes, 0, written - blob.bytes, rounds));
                }
                written = blob.bytes;
                delta.copied_bytes += blob.bytes;
            }
        }
        if (!reads.empty() || !window_reads.empty()) {
            if (direct) {
                direct->read(window_reads);
            } else {
                read_now(reads);
            }
            for (const auto& copy : staged_copies) {
                CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                           direct ? cudaMemcpyDefault : cudaMemcpyHostToDevice,
                                           rounds));
            }
            CUDA_CHECK(cudaEventRecord(round_staged, rounds));
        }
        CUDA_CHECK(cudaEventRecord(done, rounds));
        for (std::size_t i = 0; i < issued; ++i) {
            auto d = queued[i];
            d.landed = done;
            deferred.push_back(d);
        }
        queued.erase(queued.begin(), queued.begin() + std::ptrdiff_t(issued));
        if (issued == 0) { spare_events.push_back(done); }
    }

    [[nodiscard]] int bank_holding(std::size_t li) const {
        for (std::size_t b = 0; b < bank_state.size(); ++b) {
            if (bank_state[b].layer == std::int64_t(li)) { return int(b); }
        }
        return -1;
    }

    // A served call may write any bank: the preceding call's consumers finished before it was
    // published. It takes the bank the last bank-using call did not use, which keeps that layer
    // for an immediate repeat (a second prefill slice, the next draft of the same layer).
    [[nodiscard]] int pick_bank(int avoid) const {
        return bank_state.size() == 1 ? 0 : avoid >= 0 ? 1 - avoid : 1 - last_bank;
    }
    void claim_bank(int k, std::size_t li) {
        auto& bank = bank_state[std::size_t(k)];
        bank.layer = std::int64_t(li);
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }

    // Copy experts of layer li into bank k at their indices, one piece per contiguous run, and keep
    // zeros after the last expert of each run that the bank does not already continue.
    void fill_bank(int k, std::size_t li, const std::vector<std::int32_t>& experts) {
        auto& layer = layers[li];
        auto& bank = bank_state[std::size_t(k)];
        const auto blobs = layer.source.experts.front().size();
        for (std::size_t b = 0; b < blobs; ++b) {
            for (const auto e : experts) {
                add_piece(bank_blob(k, li, e, b), layer.source.experts[std::size_t(e)][b]);
            }
        }
        for (const auto e : experts) { bank.valid[std::size_t(e)] = 1; }
        for (std::size_t b = 0; b < blobs; ++b) {
            if (!tail[b]) { continue; }
            for (const auto e : experts) {
                const auto next = std::size_t(e) + 1;
                if (next < layer.source.experts.size() && bank.valid[next]) { continue; }
                const auto end = std::uint64_t(e) * layer.stride[b] +
                                 layer.source.experts[std::size_t(e)][b].bytes;
                const auto room = bank_size[b] - std::min(bank_size[b], end);
                if (room) {
                    zeroes.push_back({static_cast<std::byte*>(banks[std::size_t(k)].p) +
                                          bank_offset[b] + end,
                                      std::min(tail[b], room)});
                }
            }
        }
    }

    // Device copies of bank expert e into slot s.
    void promote(std::int32_t s, std::size_t li, std::int32_t e, int bank, ExpertDeliveryStats& delta) {
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto bytes = layers[li].source.experts[std::size_t(e)][b].bytes;
            CUDA_CHECK(cudaMemcpyAsync(slot_blob(s, b), bank_blob(bank, li, e, b), bytes,
                                       cudaMemcpyDeviceToDevice, copies));
            slot_written(s, b, bytes);
            delta.promoted_bytes += bytes;
        }
        ++delta.admissions;
    }

    // The CPU's share of a narrow call's misses that neither a slot nor the layer's bank holds:
    // the experts with the fewest pairs (a copy serves all of an expert's pairs at once, the CPU's
    // work grows with each). Marks them in call_cpu.
    std::vector<std::int32_t> cpu_share(std::size_t li, const std::uint32_t* needed, int held) {
        const auto& layer = layers[li];
        std::vector<std::pair<std::uint32_t, std::int32_t>> order;
        for (int e = 0; e < int(layer.source.experts.size()); ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            const auto s = layer.slot_of[std::size_t(e)];
            if (s >= 0 && !slots[std::size_t(s)].pending) { continue; }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) { continue; }
            // hp-tune (N11''): a read-ahead copy into a lent slot landed after the split.
            if (ahead_max && layer.lent_of[std::size_t(e)] >= 0) { continue; }
            order.emplace_back(needed[e] & kCountMask, e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t count = 0;
        if (options.cpu.balance) {
            std::vector<int> pairs;
            for (const auto& [n, e] : order) { pairs.push_back(int(n)); }
            count = split.cpu_count(pairs, megabytes(li));
        } else {
            count = std::min(order.size(),
                             std::size_t(std::lround(double(order.size()) * options.cpu.share)));
        }
        std::vector<std::int32_t> out;
        for (std::size_t i = order.size() - count; i < order.size(); ++i) {
            out.push_back(order[i].second);
            call_cpu[std::size_t(order[i].second)] = 1;
        }
        return out;
    }

    [[nodiscard]] double megabytes(std::size_t li) const {
        double bytes = 0;
        for (const auto& blob : layers[li].source.experts.front()) { bytes += double(blob.bytes); }
        return bytes / 1e6;
    }

    // The CPU's products of `cpu` for the served call of `pairs` pairs after this thread issues
    // the call's copies (`issue`): the products' setup (the activation's repacking, the jobs'
    // buffers) and a disk expert's reads would otherwise delay every copy of the call. Then its
    // missing ids without the CPU's pairs, its products and flag for the device.
    void compute_cpu(std::size_t li, const std::vector<std::int32_t>& cpu, std::uint32_t pairs,
                     const std::uint32_t* needed, ExpertDeliveryStats& delta,
                     const std::function<void()>& issue) {
        auto* flag = static_cast<std::int32_t*>(host_flag->data());
        if (cpu.empty()) {
            issue();
            if (*flag != 0) {
                *flag = 0;
                CUDA_CHECK(cudaMemcpyAsync(cpu_flag.p, flag, 4, cudaMemcpyHostToDevice, copies));
            }
            return;
        }
        const auto& layer = layers[li];
        std::vector<std::vector<const std::byte*>> blobs(cpu.size());
        std::vector<QueuedFileRead> reads;
        std::uint64_t file_bytes = 0;
        for (const auto e : cpu) {
            for (const auto& blob : layer.source.experts[std::size_t(e)]) {
                file_bytes += blob.pinned ? 0 : blob.bytes;
            }
        }
        cpu_bytes.resize(std::size_t(file_bytes));
        std::uint64_t at = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            for (const auto& blob : layer.source.experts[std::size_t(cpu[i])]) {
                if (blob.pinned) {
                    blobs[i].push_back(static_cast<const std::byte*>(blob.pinned));
                    continue;
                }
                blobs[i].push_back(cpu_bytes.data() + at);
                for (const auto& extent : blob.extents) {
                    reads.push_back({extent.file, extent.offset,
                                     {reinterpret_cast<std::uint8_t*>(cpu_bytes.data() + at),
                                      std::size_t(extent.bytes)}});
                    at += extent.bytes;
                }
            }
        }
        issue();
        if (!reads.empty()) { read_now(reads); }
        const auto* ids = static_cast<const std::int32_t*>(host_ids->data());
        ExpertCpuProducts::Work work;
        work.layer    = li;
        work.experts  = cpu;
        work.blobs    = blobs;
        work.ids      = {ids, pairs};
        work.inputs   = static_cast<const std::byte*>(host_inputs->data());
        work.products = static_cast<std::byte*>(host_products->data());
        work.pool     = pool.get();
        const auto [used, microseconds] = options.cpu.products->compute(work);
        if (probe_on) { probe_now.cpu_done = probe_clock(); }
        if (used > options.cpu.product_bytes) {
            throw std::logic_error("expert delivery: CPU products exceed their buffer");
        }
        // The missing set without the CPU's experts: per pair, then as a list.
        auto* missing = static_cast<std::int32_t*>(host_missing->data());
        auto* listed      = missing + options.pair_capacity;
        const int experts = int(layer.source.experts.size());
        const auto left   = [&](std::int32_t e) {
            return e >= 0 && e < experts && (needed[e] & kMiss) && !call_cpu[std::size_t(e)];
        };
        for (std::uint32_t p = 0; p < pairs; ++p) { missing[p] = left(ids[p]) ? ids[p] : -1; }
        listed[0] = 0;
        for (int e = 0; e < experts; ++e) {
            if (left(e)) { listed[1 + listed[0]++] = e; }
        }
        CUDA_CHECK(cudaMemcpyAsync(missing_ids.p, missing, std::size_t(pairs) * 4, cudaMemcpyHostToDevice,
                                   copies));
        CUDA_CHECK(cudaMemcpyAsync(missing_experts, listed, std::size_t(1 + listed[0]) * 4,
                                   cudaMemcpyHostToDevice, copies));
        if (used != 0) {
            CUDA_CHECK(cudaMemcpyAsync(products.p, host_products->data(), used, cudaMemcpyHostToDevice,
                                       copies));
        }
        *flag = 1;
        CUDA_CHECK(cudaMemcpyAsync(cpu_flag.p, flag, 4, cudaMemcpyHostToDevice, copies));
        double work_done = 0;
        for (const auto e : cpu) {
            const int n = int(needed[e] & kCountMask);
            delta.cpu_pairs += std::uint64_t(n);
            work_done += ExpertSplit::cpu_work(megabytes(li), n);
        }
        delta.cpu_experts += cpu.size();
        delta.read_bytes += file_bytes;
        if (options.cpu.balance) { split.computed(work_done, microseconds); }
    }

    void serve(const CudaHostHandshake::Request& request) {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        const std::size_t li = request.tag;
        auto& layer = layers.at(li);
        call_layer = li;
        const auto* needed = static_cast<const std::uint32_t*>(needed_host->data());
        const int experts = int(layer.source.experts.size());
        delta.requests = 1;
        std::uint32_t routes = 0;
        std::fill(call_routed.begin(), call_routed.end(), 0);
        std::fill(call_cpu.begin(), call_cpu.end(), 0);
        std::uint32_t dropped = 0;
        for (int e = 0; e < experts; ++e) {
            const std::uint32_t count = needed[e] & kCountMask;
            call_count[std::size_t(e)] = count;
            if (needed[e] & kSkip) { ++dropped; }
            if (!count) { continue; }
            ++routes;
            call_routed[std::size_t(e)] = 1;
            delta.pairs += count;
            if (needed[e] & kHit) {
                ++delta.hits;
                delta.pair_hits += count;
            }
        }
        delta.routes = routes;
        const bool wide = 2 * routes >= std::uint32_t(experts);
        if (probe_on) {
            probe_now          = {};
            probe_now.sequence = request.sequence;
            probe_now.layer    = std::uint32_t(li);
            probe_now.routes   = routes;
            probe_now.hits     = std::uint32_t(delta.hits);
            probe_now.wide     = wide;
            probe_now.seen     = probe_clock(started);
        }
        if (guess_host) {
            // hp-tune (N11''): the guess of this layer is scored and cleared.
            if (guess_score && !wide) { count_guess(li, needed); }
            const std::size_t stride = 1 + std::size_t(guess_rows) * guess_width;
            auto* area = static_cast<volatile std::int32_t*>(guess_host->data());
            area[li * stride] = 0;
            if (ahead_max) {
                // hp-tune (N11''): the next request's layer is the one whose guess is here now.
                ExpertAheadStats d;
                if (ahead_expect >= 0 && ahead_expect != std::int64_t(li)) {
                    ++d.mismatch;
                    if (ahead_check) {
                        std::fprintf(stderr, "hp-prefetch CHECK: read ahead for layer %lld, the "
                                             "next call is layer %zu\n",
                                     (long long)ahead_expect, li);
                    }
                }
                ahead_expect = ahead_target = -1;
                ahead_rows   = 0;
                if (!wide) {
                    int found = 0;
                    for (std::size_t l = 0; l < layers.size(); ++l) {
                        const std::int32_t n = area[l * stride];
                        if (l == li || n <= 0 || std::uint32_t(n) > guess_rows) { continue; }
                        ++found;
                        ahead_target = std::int64_t(l);
                        ahead_rows   = n;
                    }
                    if (found > 1) {
                        ++d.ambiguous;
                        ahead_target = -1;
                    }
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
                ahead_add(d);
            }
        }
        if (skip_on && !wide) {
            // hp-tune (miss-skip): the dropped experts were misses and routes before the split.
            if (li == 0 && ++skip_passes % 256 == 0) {
                std::fprintf(stderr,
                             "hp-tune: miss-skip after %llu passes: %llu dropped / %llu missing / "
                             "%llu routes (%.1f%% of misses, %.2f%% of routes)\n",
                             (unsigned long long)skip_passes, (unsigned long long)skip_dropped,
                             (unsigned long long)skip_missing, (unsigned long long)skip_routes,
                             100.0 * double(skip_dropped) / double(std::max<std::uint64_t>(skip_missing, 1)),
                             100.0 * double(skip_dropped) / double(std::max<std::uint64_t>(skip_routes, 1)));
            }
            skip_dropped += dropped;
            skip_routes += routes + dropped;
            skip_missing += dropped;
            for (int e = 0; e < experts; ++e) { skip_missing += (needed[e] & kMiss) ? 1 : 0; }
        }
        land(!wide);
        ExpertAheadStats ahead_delta; // hp-tune (N11'')
        if (ahead_max) { ahead_settle(li, needed, wide, ahead_delta); }
        ++stamp;
        for (int e = 0; e < experts; ++e) {
            const auto l = layer.lent_of[std::size_t(e)];
            if (call_routed[std::size_t(e)] && l >= 0) { lent_slots[std::size_t(l)].used = stamp; }
        }
        // Only wide calls in a row to two layers set a successor: the narrow calls between prompts
        // and a prompt span's repeated calls to one layer leave the order of its passes.
        if (wide && previous_wide && previous_layer >= 0 && previous_layer != std::int64_t(li)) {
            layers[std::size_t(previous_layer)].successor = std::int64_t(li);
        }
        previous_layer = std::int64_t(li);
        previous_wide  = wide;
        const bool new_pass = !wide && last_narrow >= 0 && std::int64_t(li) < last_narrow;
        loop_begin(li, wide); // hp-tune (N11/N12)
        if (wide) { quiet_passes = 0; }
        if (per_bank && new_pass && ++quiet_passes >= 2) {
            // Only narrow calls ran for two passes: the banks become cache slots.
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (!bank_lent[b]) { lend_bank(int(b)); }
            }
        }
        const int held = bank_holding(li);
        int k = held;
        if (wide && per_bank) {
            // The wide call's bank: a lent one is taken back first.
            if (k < 0) { k = pick_bank(-1); }
            if (bank_lent[std::size_t(k)]) { reclaim_bank(k, delta); }
        }
        const bool lent = !wide && any_lent();
        std::vector<std::int32_t> lent_taken;
        const auto cpu = !wide && layer.cpu && (request.flags & kCpuCall) ? cpu_share(li, needed, held)
                                                                          : std::vector<std::int32_t>{};
        if (!cpu.empty()) { cpu_until = started + std::chrono::milliseconds(500); }
        std::vector<std::int32_t> to_bank;
        for (int e = 0; e < experts; ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            ++delta.misses;
            if (call_cpu[std::size_t(e)]) { continue; }
            const auto s = layer.slot_of[std::size_t(e)];
            if (s >= 0 && !slots[std::size_t(s)].pending) {
                // Promoted or landed after the split read its entry.
                point_slot(li, e, s);
                continue;
            }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) {
                ++delta.bank_reuses;
                point_bank(li, e, held);
                continue;
            }
            // A wide call reads a lent slot's entry as a miss; the slot keeps its bytes until a
            // later call takes its bank back.
            if (layer.lent_of[std::size_t(e)] >= 0) { continue; }
            // A round's candidate whose copy is still in flight is served from the bank.
            if (!wide && s < 0) {
                const auto slot = admit_slot(li, e, needed[e] & kCountMask, delta);
                if (slot >= 0) {
                    occupy(slot, li, e, false);
                    for (std::size_t b = 0; b < layer.source.experts[std::size_t(e)].size(); ++b) {
                        const auto& blob = layer.source.experts[std::size_t(e)][b];
                        add_piece(slot_blob(slot, b), blob);
                        slot_written(slot, b, blob.bytes);
                    }
                    ++delta.admissions;
                    ++probe_now.copied;
                    point_slot(li, e, slot);
                    continue;
                }
            }
            if (lent) {
                // Lent banks take every miss no cache slot took, so none needs a bank.
                const auto l = next_lent(li, lent_taken);
                if (l < 0) {
                    throw std::logic_error("expert delivery: no lent slot for a narrow call's miss");
                }
                lent_taken.push_back(l);
                admit_lent(l, li, e, delta);
                ++probe_now.copied;
                continue;
            }
            to_bank.push_back(e);
        }
        loop_end_call(li, needed); // hp-tune (N11)
        // The tier learns from decode and verification only; a call below the last narrow one
        // starts a pass (a draft layer's repeated calls do not). It learns once the call's copies
        // are issued, so a pass's round does not hold them back; the call's tables, uploaded
        // after it, carry the round's evictions before any later layer runs.
        const auto learn = [&] {
            if (wide) { return; }
            if (new_pass) { round(delta); }
            last_narrow = std::int64_t(li);
            std::lock_guard lock(mutex);
            for (int e = 0; e < experts; ++e) {
                const auto count = call_count[std::size_t(e)];
                if (!count) { continue; }
                tier->route(li, e, float(count));
                layer.total[std::size_t(e)] += count;
            }
        };
        if (!to_bank.empty()) {
            if (k < 0) { k = pick_bank(-1); }
            if (k != held) { claim_bank(k, li); }
        }
        for (const auto e : to_bank) { point_bank(li, e, k); }
        if (k >= 0 && (held >= 0 || !to_bank.empty())) { last_bank = k; }
        call_bank = (held >= 0 || !to_bank.empty()) ? k : -1;
        call_wide = wide;
        if (!to_bank.empty()) { fill_bank(k, li, to_bank); }
        const bool timed = options.cpu.balance && !cpu.empty() && !pieces.empty();
        const auto copied_before = delta.copied_bytes;

        const auto issue = [&] {
            if (timed) { CUDA_CHECK(cudaEventRecord(copies_begun, copies)); }
            flush_pieces(delta);
            if (timed) { CUDA_CHECK(cudaEventRecord(copies_done, copies)); }
            if (probe_on) { probe_now.issued = probe_now.cpu_done = probe_clock(); }
            learn();
        };
        if (layer.cpu && host_flag) {
            compute_cpu(li, cpu, request.width, needed, delta, issue);
        } else {
            issue();
        }
        if (ranges_dirty) {
            CUDA_CHECK(cudaMemcpyAsync(lent_ranges.p, host_ranges->data(), host_ranges->size(),
                                       cudaMemcpyHostToDevice, copies));
            ranges_dirty = false;
        }
        if (ahead_check) { ahead_verify(li); }
        upload_tables();
        const auto waiting = std::chrono::steady_clock::now();
        CUDA_CHECK(ninfer::wait_for_stream(copies));
        if (ahead_max && !wide) {
            const auto now = std::chrono::steady_clock::now();
            ahead_delta.narrow   = 1;
            ahead_delta.wait_ns  = std::uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - waiting).count());
            ahead_delta.serve_ns = std::uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - started).count());
        }
        if (ahead_max) { ahead_add(ahead_delta); }
        if (probe_on) {
            probe_now.copies_done = probe_clock();
            probe_now.copied += std::uint32_t(to_bank.size());
            probe_now.misses = std::uint32_t(delta.misses);
            probe_now.cpu    = std::uint32_t(cpu.size());
            probe_now.bytes  = delta.copied_bytes - copied_before;
        }
        if (timed) {
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, copies_begun, copies_done));
            split.copied(double(delta.copied_bytes - copied_before) / 1e6, 1e3 * double(milliseconds));
        }
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    // After the call's experts are ready: a wide call promotes its most routed bank experts into
    // free cache slots by device copies, up to the layer's share of the cache, so a first prompt
    // fills the cache across every layer and evicts nothing. It also starts the bank of the layer
    // that followed it last time.
    void post() {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        if (!call_wide) {
            if (queued.empty()) { return; }
            issue_rounds(delta);
            delta.service_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            add(delta);
            return;
        }
        const std::size_t li = call_layer;
        auto& layer = layers[li];
        if (pool_of(li).count != 0 && call_bank >= 0) {
            const auto& bank = bank_state[std::size_t(call_bank)];
            std::vector<std::int32_t> candidates;
            for (std::size_t e = 0; e < layer.source.experts.size(); ++e) {
                if (call_routed[e] && bank.valid[e] && layer.slot_of[e] < 0) {
                    candidates.push_back(std::int32_t(e));
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](std::int32_t a, std::int32_t b) {
                const auto ca = call_count[std::size_t(a)], cb = call_count[std::size_t(b)];
                return ca != cb ? ca > cb : a < b;
            });
            auto& pool       = pool_of(li);
            const auto share = std::max<std::size_t>(1, pool.count / pool.layers);
            for (const auto e : candidates) {
                if (pool.free.empty() || layer.cached >= share) { break; }
                const auto s = pool.free.back();
                pool.free.pop_back();
                occupy(s, li, e, true);
                promote(s, li, e, call_bank, delta);
                deferred.push_back({li, e, s});
            }
        }
        if (bank_state.size() == 2 && layer.successor >= 0) {
            const auto next = std::size_t(layer.successor);
            const bool held = std::any_of(bank_state.begin(), bank_state.end(),
                                          [&](const Bank& b) { return b.layer == std::int64_t(next); });
            std::vector<std::int32_t> experts;
            for (std::size_t e = 0; e < layers[next].source.experts.size() && !held; ++e) {
                if (layers[next].slot_of[e] < 0) { experts.push_back(std::int32_t(e)); }
            }
            // Never the bank this call's consumers are reading, nor a lent one: the next call may
            // be narrow and read its slots.
            const int k = pick_bank(call_bank);
            if (!experts.empty() && !bank_lent[std::size_t(k)]) {
                claim_bank(k, next);
                fill_bank(k, next, experts);
                ++delta.prefetched_layers;
            }
        }
        flush_pieces(delta); // tables are unchanged: promoted entries are patched at the next request
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    void add(const ExpertDeliveryStats& d) {
        std::lock_guard lock(mutex);
        counters = counters + d;
    }

    void fail(const CudaHostHandshake::Request* request) {
        {
            std::lock_guard lock(mutex);
            if (!failure) { failure = std::current_exception(); }
        }
        batch_reads.clear();
        batch_copies.clear();
        pieces.clear();
        zeroes.clear();
        staging_cursor = 0;
        if (request) {
            // Captured decode can finish safely; the caller checks before publishing output.
            cudaMemsetAsync(missing_ids.p, 0xff, std::size_t(request->width) * 4, copies);
            cudaMemsetAsync(missing_experts, 0, 4, copies);
            if (cpu_flag.bytes) {
                cudaMemsetAsync(cpu_flag.p, 0, 4, copies);
                *static_cast<std::int32_t*>(host_flag->data()) = 0;
            }
        }
        ninfer::wait_for_stream(copies);
    }

    void run() {
        DeviceBinding bind(rank.device);
        std::uint32_t served = 0;
        auto active_until = std::chrono::steady_clock::now();
        while (!stopping.load(std::memory_order_acquire)) {
            const auto request = handshake->pending(served);
            const auto now     = std::chrono::steady_clock::now();
            if (spinning && now >= cpu_until) {
                pool->spin(false);
                spinning = false;
            }
            if (!request) {
                if (now < active_until) { std::this_thread::yield(); }
                else { std::this_thread::sleep_for(std::chrono::microseconds(20)); }
                continue;
            }
            if (pool && !spinning && now < cpu_until) {
                pool->spin(true);
                spinning = true;
            }
            bool served_ok = false;
            try { check(); serve(*request); served_ok = true; }
            catch (...) { fail(&*request); }
            if (probe_on && served_ok) {
                // hp-tune (probe): published before the ready signal, so a reader after the
                // compute stream's last call finds every call's record.
                probe_now.prepared = probe_clock();
                const std::uint64_t n = probe_count.load(std::memory_order_relaxed);
                probe_ring[std::size_t(n % kProbeRing)] = probe_now;
                probe_count.store(n + 1, std::memory_order_release);
            }
            handshake->prepared(request->sequence);
            served = request->sequence;
            if (served_ok && ahead_max) {
                // hp-tune (N11''): the next visit's read-ahead before the round's copies.
                try { ahead_issue(); }
                catch (...) { ahead_target = -1; fail(nullptr); }
            }
            if (served_ok) {
                try { post(); }
                catch (...) { fail(nullptr); }
            }
            finished.store(served, std::memory_order_release);
            active_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        }
    }

    // After the compute stream is idle: the worker has finished every published request,
    // post work included, and its copies are done.
    void quiesce() {
        DeviceBinding bind(rank.device);
        CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        while (const auto request = handshake->pending(finished.load(std::memory_order_acquire))) {
            (void)request;
            std::this_thread::yield();
        }
        CUDA_CHECK(ninfer::wait_for_stream(copies));
        CUDA_CHECK(ninfer::wait_for_stream(rounds));
        if (ahead_stream) { CUDA_CHECK(ninfer::wait_for_stream(ahead_stream)); }
    }

    // An elastic cache, quiesced and locked: gives back at least `bytes` of whole granules, or
    // every slot, taking slots from the pools whose layers hold the most each (trim_pools). Each
    // pool keeps the most routed experts it holds (the tier's decayed counts, then the routes since
    // prime()), moved into its remaining slots when they sat past them; the others leave, and so
    // do a round's candidates whose copies were not issued yet.
    std::uint64_t shrink(std::uint64_t bytes) {
        const std::uint64_t before = cache.growable.mapped_bytes();
        if (cache_slots() == 0 || bytes == 0) { return 0; }
        const std::uint64_t g    = cache.growable.granule();
        const std::uint64_t drop = (std::min(bytes, before) + g - 1) / g * g;
        std::vector<std::uint32_t> counts;
        for (const auto& pool : pools) { counts.push_back(pool.count); }
        const auto keep = trim_pools(before - std::min(before, drop), counts);
        ExpertDeliveryStats delta;
        ExpertAheadStats ahead_delta; // hp-tune (N11''): every read-ahead entry moves first
        ahead_flush(ahead_delta);
        ahead_counts.wasted += ahead_delta.wasted;
        land();
        for (const auto& d : queued) {
            if (slots[std::size_t(d.slot)].layer >= 0) { evict(d.slot, delta); }
        }
        queued.clear();
        const auto rank_of = [&](std::int32_t s) {
            const auto& slot = slots[std::size_t(s)];
            return std::pair{tier->count(std::size_t(slot.layer), slot.expert),
                             layers[std::size_t(slot.layer)].total[std::size_t(slot.expert)]};
        };
        for (std::size_t p = 0; p < pools.size(); ++p) {
            auto& pool     = pools[p];
            const auto end = pool.first + std::int32_t(keep[p]);
            std::vector<std::int32_t> held;
            for (std::int32_t s = pool.first; s < pool.first + std::int32_t(pool.count); ++s) {
                if (slots[std::size_t(s)].layer >= 0) { held.push_back(s); }
            }
            std::stable_sort(held.begin(), held.end(), [&](std::int32_t a, std::int32_t b) {
                return rank_of(a) > rank_of(b);
            });
            for (std::size_t i = keep[p]; i < held.size(); ++i) { evict(held[i], delta); }
            held.resize(std::min<std::size_t>(held.size(), keep[p]));
            std::vector<std::int32_t> inside;
            for (std::int32_t s = end - 1; s >= pool.first; --s) {
                if (slots[std::size_t(s)].layer < 0) { inside.push_back(s); }
            }
            for (const auto from : held) {
                if (from < end) { continue; }
                const auto to = inside.back();
                inside.pop_back();
                CUDA_CHECK(cudaMemcpyAsync(slot_base(to), slot_base(from), pool.slot_bytes,
                                           cudaMemcpyDeviceToDevice, copies));
                auto& source   = slots[std::size_t(from)];
                auto& target   = slots[std::size_t(to)];
                target.layer   = source.layer;
                target.expert  = source.expert;
                target.pending = false;
                target.written = source.written;
                layers[std::size_t(source.layer)].slot_of[std::size_t(source.expert)] = to;
                point_slot(std::size_t(source.layer), source.expert, to);
                source.layer = source.expert = -1;
            }
            pool.free  = std::move(inside);
            pool.count = keep[p];
        }
        upload_tables();
        CUDA_CHECK(ninfer::wait_for_stream(copies));
        cache.bytes = 0;
        for (const auto& pool : pools) {
            const std::uint64_t end = std::uint64_t(pool.count) * pool.slot_bytes;
            cache.growable.unmap(pool.base + end, pool.span - end);
            cache.bytes += backing(pool, pool.count);
        }
        set_free();
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = cache_slots();
        counters = counters + delta;
        return before - cache.growable.mapped_bytes();
    }

    // An elastic cache, quiesced and locked: empty slots in the free device memory beyond
    // keep_free_bytes, up to every expert (or an explicit cache size), as much of it as the device
    // supplies, to the pools whose layers hold the fewest each (fill_pools).
    std::uint64_t grow() {
        const std::uint64_t before = cache.growable.mapped_bytes();
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const std::uint64_t g    = cache.growable.granule();
        const std::uint64_t room = free > options.keep_free_bytes
                                       ? (free - options.keep_free_bytes) / g * g : 0;
        std::uint64_t budget     = before + room;
        if (options.cache_bytes) { budget = std::min(budget, *options.cache_bytes); }
        std::vector<std::uint32_t> counts;
        for (const auto& pool : pools) { counts.push_back(pool.count); }
        const auto target = fill_pools(budget, counts);
        for (std::size_t p = 0; p < pools.size(); ++p) {
            auto& pool     = pools[p];
            const auto old = pool.count;
            auto want      = target[p];
            // The slots that fit in the granules mapped now and in `room` more.
            while (want > old &&
                   !cache.growable.map(pool.base + std::uint64_t(old) * pool.slot_bytes,
                                       std::uint64_t(want - old) * pool.slot_bytes)) {
                want = old + (want - old) / 2;
            }
            if (want <= old) { continue; }
            CUDA_CHECK(cudaMemsetAsync(slot_base(pool.first + std::int32_t(old)), 0,
                                       std::uint64_t(want - old) * pool.slot_bytes, copies));
            for (auto s = pool.first + std::int32_t(old); s < pool.first + std::int32_t(want);
                 ++s) {
                std::fill(slots[std::size_t(s)].written.begin(),
                          slots[std::size_t(s)].written.end(), 0);
                pool.free.push_back(s);
            }
            std::sort(pool.free.begin(), pool.free.end(), std::greater<>());
            pool.count = want;
        }
        CUDA_CHECK(ninfer::wait_for_stream(copies));
        cache.bytes = 0;
        for (const auto& pool : pools) { cache.bytes += backing(pool, pool.count); }
        set_free();
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = cache_slots();
        return cache.growable.mapped_bytes() - before;
    }
};

ExpertDelivery::ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                               std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options)
    : impl_(std::make_unique<Impl>(rank, std::move(files), std::move(layers), options)) {}
ExpertDelivery::~ExpertDelivery() = default;

ExpertDelivery::Call ExpertDelivery::begin(std::size_t index, const std::int32_t* ids,
                                           std::uint32_t pairs, cudaStream_t stream,
                                           std::span<const Input> inputs, Skip skip) {
    auto& p = *impl_;
    p.check();
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    const auto& layer = p.layers.at(index);
    if (!ids || !pairs || pairs > p.options.pair_capacity || stream != p.rank.stream) {
        throw std::invalid_argument("expert delivery: invalid call or compute stream");
    }
    const auto low = reinterpret_cast<std::uintptr_t>(p.cache.p);
    const int experts = int(layer.source.experts.size());
    Call call{static_cast<const std::int32_t*>(p.cached_ids.p),
              static_cast<const std::int32_t*>(p.missing_ids.p), p.cached_experts,
              p.missing_experts};
    const bool cpu = p.host_flag && layer.cpu && !inputs.empty();
    HostCopies copies;
    if (cpu) {
        // The CPU's operands: the pairs' experts and the family's inputs.
        if (inputs.size() > std::size_t(kCpuInputs)) {
            throw std::invalid_argument("expert delivery: more than three CPU inputs");
        }
        copies.source[0] = reinterpret_cast<const std::byte*>(ids);
        copies.target[0] = static_cast<std::byte*>(p.host_ids_mapped);
        copies.bytes[0]  = pairs * 4;
        std::size_t at = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (input.bytes > p.options.cpu.input_bytes - std::min(at, p.options.cpu.input_bytes)) {
                throw std::invalid_argument("expert delivery: CPU inputs exceed their buffer");
            }
            copies.source[1 + i] = static_cast<const std::byte*>(input.data);
            copies.target[1 + i] = static_cast<std::byte*>(p.host_inputs_mapped) + at;
            copies.bytes[1 + i]  = std::uint32_t(input.bytes);
            at += input.bytes;
        }
        call.products = p.products.p;
        call.cpu = static_cast<const std::int32_t*>(p.cpu_flag.p);
    }
    // hp-tune (miss-skip): only when enabled and the caller gave whole rows of weights.
    SplitSkip split_skip = p.skip_limits;
    if (p.skip_on && skip.weights && skip.top_k > 0 && pairs % std::uint32_t(skip.top_k) == 0) {
        split_skip.weights = skip.weights;
        split_skip.scale   = skip.scale;
        split_skip.top_k   = skip.top_k;
    }
    split_routes<<<1, 1024, std::size_t(experts) * 4, stream>>>(
        ids, pairs, experts, static_cast<const void* const*>(layer.tables.p), low,
        low + p.cache_span, static_cast<const std::uintptr_t*>(p.lent_ranges.p),
        static_cast<std::int32_t*>(p.cached_ids.p), static_cast<std::int32_t*>(p.missing_ids.p),
        p.cached_experts, p.missing_experts, static_cast<std::uint32_t*>(p.needed_mapped), copies,
        p.handshake->device(), std::uint32_t(index), cpu ? kCpuCall : 0, split_skip);
    CUDA_CHECK(cudaGetLastError());
    return call;
}
bool ExpertDelivery::miss_skip() const noexcept { return impl_->skip_on; }
void ExpertDelivery::await(cudaStream_t stream) {
    impl_->handshake->wait_prepared(stream);
    impl_->check();
}
const void* const* ExpertDelivery::tables(std::size_t layer) const {
    return static_cast<const void* const*>(impl_->layers.at(layer).tables.p);
}
bool ExpertDelivery::alias_aligned(std::size_t layer, std::size_t alias,
                                   std::uint64_t bytes) const {
    if (bytes == 0 || bytes > 256 || (bytes & (bytes - 1)) != 0) {
        throw std::invalid_argument("expert delivery: an alignment is a power of two up to 256");
    }
    const auto& l = impl_->layers.at(layer);
    const auto& a = l.source.aliases.at(alias);
    return a.offset % bytes == 0 && l.stride.at(a.blob) % bytes == 0;
}
void ExpertDelivery::check() const { impl_->check(); }
void ExpertDelivery::drain() {
    impl_->quiesce();
    impl_->check();
}
void ExpertDelivery::invalidate() {
    auto& p = *impl_;
    if (p.released) { throw std::logic_error("expert delivery: restore backing before invalidation"); }
    drain();
    for (auto& layer : p.layers) {
        layer.tables.fill();
        std::memset(layer.mirror->data(), 0, layer.mirror->size());
        std::fill(layer.slot_of.begin(), layer.slot_of.end(), -1);
        layer.cached = 0;
        layer.dirty = false;
    }
    for (auto& slot : p.slots) {
        slot.layer = slot.expert = -1;
        slot.pending = false;
    }
    for (auto& pool : p.pools) { Impl::reset_free(pool); }
    for (auto& bank : p.bank_state) {
        bank.layer = -1;
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }
    // No bank stays lent: the split reads an empty range for each.
    for (auto& layer : p.layers) {
        std::fill(layer.lent_of.begin(), layer.lent_of.end(), -1);
        std::fill(layer.bank_of.begin(), layer.bank_of.end(), std::int8_t(-1));
    }
    for (auto& slot : p.lent_slots) {
        slot.layer = slot.expert = -1;
        slot.ahead = false;
    }
    // hp-tune (N11''): drained, so every read-ahead copy has landed; none is judged.
    for (const auto& r : p.ahead) { p.spare_events.push_back(r.landed); }
    p.ahead.clear();
    p.ahead_target = p.ahead_expect = -1;
    std::fill(p.bank_lent.begin(), p.bank_lent.end(), 0);
    std::memset(p.host_ranges->data(), 0, p.host_ranges->size());
    p.lent_ranges.fill();
    p.ranges_dirty = false;
    p.quiet_passes = 0;
    p.set_lent_count();
    for (const auto& d : p.deferred) {
        if (d.landed && std::find(p.spare_events.begin(), p.spare_events.end(), d.landed) ==
                            p.spare_events.end()) {
            p.spare_events.push_back(d.landed);
        }
    }
    p.deferred.clear();
    p.queued.clear();
    for (std::size_t li = 0; li < p.layers.size(); ++li) {
        for (std::size_t e = 0; e < p.layers[li].source.experts.size(); ++e) {
            p.tier->set(li, std::int32_t(e), ExpertTier::State::Out);
        }
    }
    p.set_free();
    p.previous_layer = -1;
    p.previous_wide  = false;
    p.last_narrow = -1;
    if (p.host_flag) {
        *static_cast<std::int32_t*>(p.host_flag->data()) = 0;
        p.cpu_flag.fill();
    }
}
void ExpertDelivery::release_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (p.released) { return; }
    drain();
    DeviceBinding bind(p.rank.device);
    p.each_storage([](Storage& storage) { storage.release_backing(); });
    p.released = true;
}
void ExpertDelivery::restore_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (!p.released) { return; }
    DeviceBinding bind(p.rank.device);
    try {
        p.each_storage([](Storage& storage) { storage.restore_backing(); });
        p.each_storage([](Storage& storage) { storage.fill(); });
        p.handshake->resume();
        for (auto& slot : p.slots) { std::fill(slot.written.begin(), slot.written.end(), 0); }
        p.released = false;
        invalidate();
    } catch (...) {
        p.each_storage([](Storage& storage) { storage.release_backing(); });
        p.released = true;
        throw;
    }
}
std::uint64_t ExpertDelivery::state_backing_bytes() const noexcept {
    const auto memory = this->memory();
    return memory.cache_bytes + memory.bank_bytes + memory.metadata_bytes;
}
ExpertDeliveryStats ExpertDelivery::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
ExpertDeliveryMemory ExpertDelivery::memory() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->allocation;
}

void ExpertDelivery::prime(const std::vector<std::vector<std::uint64_t>>* profile) {
    auto& p = *impl_;
    invalidate();
    p.tier->reset();
    {
        std::lock_guard lock(p.mutex);
        for (auto& layer : p.layers) { std::fill(layer.total.begin(), layer.total.end(), 0); }
    }
    p.set_free();
    if (profile == nullptr) { return; }
    const auto& counts = *profile;
    if (counts.size() != p.layers.size()) {
        throw std::invalid_argument("expert delivery: the profile describes other layers");
    }
    // Most routed first over every layer, each into its pool while it has slots; ties keep layer
    // and expert order.
    std::vector<std::tuple<std::uint64_t, std::size_t, std::int32_t>> ranked;
    for (std::size_t li = 0; li < p.layers.size(); ++li) {
        if (counts[li].size() != p.layers[li].source.experts.size()) {
            throw std::invalid_argument("expert delivery: the profile's bank width differs");
        }
        for (std::size_t e = 0; e < counts[li].size(); ++e) {
            if (counts[li][e] != 0 && p.layers[li].slot_of[e] < 0) {
                ranked.emplace_back(counts[li][e], li, std::int32_t(e));
            }
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    DeviceBinding bind(p.rank.device);
    // The service is idle before the first call; the lock orders these changes before it serves.
    std::lock_guard lock(p.mutex);
    ExpertDeliveryStats delta;
    for (const auto& [count, li, e] : ranked) {
        auto& free = p.pool_of(li).free;
        if (free.empty()) { continue; }
        const auto s = free.back();
        free.pop_back();
        p.occupy(s, li, e, false);
        for (std::size_t b = 0; b < p.layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto& blob = p.layers[li].source.experts[std::size_t(e)][b];
            p.add_piece(p.slot_blob(s, b), blob);
            p.slot_written(s, b, blob.bytes);
        }
        p.point_slot(li, e, s);
        ++delta.admissions;
    }
    p.flush_pieces(delta);
    p.upload_tables();
    CUDA_CHECK(ninfer::wait_for_stream(p.copies));
    p.set_free();
    p.counters = p.counters + delta;
}

std::vector<std::vector<std::uint64_t>> ExpertDelivery::totals() const {
    std::lock_guard lock(impl_->mutex);
    std::vector<std::vector<std::uint64_t>> out;
    for (const auto& layer : impl_->layers) { out.push_back(layer.total); }
    return out;
}

std::uint64_t ExpertDelivery::shrink_cache(std::uint64_t bytes) {
    auto& p = *impl_;
    if (!p.options.elastic) { throw std::logic_error("expert delivery: the cache is not elastic"); }
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    drain();
    DeviceBinding bind(p.rank.device);
    std::lock_guard lock(p.mutex);
    return p.shrink(bytes);
}

std::uint64_t ExpertDelivery::grow_cache() {
    auto& p = *impl_;
    if (!p.options.elastic) { throw std::logic_error("expert delivery: the cache is not elastic"); }
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    drain();
    DeviceBinding bind(p.rank.device);
    std::lock_guard lock(p.mutex);
    return p.grow();
}

void ExpertDelivery::set_keep_free_bytes(std::uint64_t bytes) {
    std::lock_guard lock(impl_->mutex);
    impl_->options.keep_free_bytes = bytes;
}

void ExpertDelivery::probe(bool on) {
    auto& p = *impl_;
    if (on && p.probe_ring.empty()) { p.probe_ring.resize(Impl::kProbeRing); }
    p.probe_on = on;
}
std::size_t ExpertDelivery::probe_records(std::span<ExpertProbeRecord> out) const {
    const auto& p        = *impl_;
    const std::uint64_t n = p.probe_count.load(std::memory_order_acquire);
    const std::uint64_t k = std::min<std::uint64_t>({n, out.size(), Impl::kProbeRing});
    for (std::uint64_t i = 0; i < k; ++i) {
        out[std::size_t(i)] = p.probe_ring[std::size_t((n - k + i) % Impl::kProbeRing)];
    }
    return std::size_t(k);
}
void ExpertDelivery::enable_guess(std::uint32_t rows, std::uint32_t width, bool score) {
    auto& p = *impl_;
    p.guess_score = score;
    if (rows == 0 || width == 0) { throw std::invalid_argument("expert delivery: an empty guess"); }
    const std::size_t words = p.layers.size() * (1 + std::size_t(rows) * width);
    p.guess_host            = std::make_unique<PinnedHostBuffer>(words * 4);
    std::memset(p.guess_host->data(), 0, words * 4);
    CUDA_CHECK(cudaHostGetDevicePointer(&p.guess_mapped, p.guess_host->data(), 0));
    p.guess_rows  = rows;
    p.guess_width = width;
}
std::int32_t* ExpertDelivery::guess_slot(std::size_t layer) const {
    const auto& p = *impl_;
    if (!p.guess_host || layer >= p.layers.size()) { return nullptr; }
    return static_cast<std::int32_t*>(p.guess_mapped) +
           layer * (1 + std::size_t(p.guess_rows) * p.guess_width);
}
ExpertGuessStats ExpertDelivery::guess_stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->guess_counts;
}
void ExpertDelivery::enable_ahead(std::uint32_t per_visit, std::uint32_t top) {
    auto& p = *impl_;
    if (!p.guess_host) { throw std::logic_error("expert delivery: read-ahead needs a guess area"); }
    if (per_visit == 0 || top == 0) { throw std::invalid_argument("expert delivery: an empty read-ahead"); }
    DeviceBinding bind(p.rank.device);
    if (!p.ahead_stream) {
        int least = 0, greatest = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&p.ahead_stream, cudaStreamNonBlocking, least));
    }
    p.ahead_check = [] {
        const char* v = std::getenv("NINFER_INK_PREFETCH_CHECK");
        return v != nullptr && std::atoi(v) == 1;
    }();
    p.ahead_sync = [] {
        const char* v = std::getenv("NINFER_INK_PREFETCH_SYNC");
        return v != nullptr && std::atoi(v) == 1;
    }();
    if (p.ahead_check || p.ahead_sync) {
        std::fprintf(stderr, "hp-tune: prefetch debug: check=%d sync=%d\n", int(p.ahead_check),
                     int(p.ahead_sync));
    }
    p.ahead_top = std::min(top, p.guess_width);
    p.ahead_max = per_visit;
}
ExpertAheadStats ExpertDelivery::ahead_stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->ahead_counts;
}
std::string ExpertDelivery::cpu_description() const {
    if (!impl_->pool) { return {}; }
    return std::to_string(impl_->pool->threads()) +
           (impl_->pool->pinned() ? " pinned threads" : " threads");
}

} // namespace ninfer
