// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

#include "thread_context.h"
using namespace shared;

#define DUMP 0

void stage1_kernel(tensor<float, loc_t::device> &Hidden_in, /* [1, 384, 8192] */
                   tensor<float, loc_t::device> &W,         /* [8192, 32000] */
                   tensor<float, loc_t::device> &Output /* [1, 384, 32000] */
) {
    thread_context ctx(bid, tid);
    tensor<float> hidden_in({1, 48, 2048});
    tensor<float> w({2048, 32000});

    tdma_load_async(hidden_in,
                    Hidden_in({0, 48 * bid, 2048 * tid}, {1, 48, 2048}), ctx);
    tdma_load_async(w, W({2048 * tid, 0}, {2048, 32000}), ctx);
    tensor_block_mma_sync(hidden_in, w, output, false, ctx);

    auto output_1 = output({0, 0, 8000 * tid}, {1, 48, 8000});
    tdma_store_async(output_1, Output({0, 48 * bid, 8000 * tid}, {1, 48, 8000}),
                     ctx);
}
