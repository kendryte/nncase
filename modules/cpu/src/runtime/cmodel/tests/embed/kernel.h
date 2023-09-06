// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

#include "thread_context.h"
using namespace shared;

#define DUMP 0

void stage1_kernel(
    [[maybe_unused]] tensor<int64_t, loc_t::device> &Position_ids, /* [1, 384] */
    [[maybe_unused]] tensor<float, loc_t::device> &GatherData,   /* [32000, 8192] */
    [[maybe_unused]] tensor<float, loc_t::device> &Output          /* [1, 384, 8192] */
) {
    thread_context ctx(bid, tid);
    tensor<int64_t> position_ids({1, 384});
    tensor<float> gather_data({384, 256});
    tensor<float> output({1, 384, 256});

    tdma_load_async(position_ids, std::move(Position_ids), ctx);
    tdma_load_async(gather_data,
                    GatherData({0, 256 * (CORES * bid + tid)}, {384, 256}),
                    ctx);
    gather(gather_data, position_ids, output, 0);

    tdma_store_async(
        output, Output({0, 0, 256 * (CORES * bid + tid)}, {1, 384, 256}), ctx);
}
