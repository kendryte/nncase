// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

#include "thread_context.h"
using namespace shared;

#define DUMP 0

void stage1_kernel(tensor<float, loc_t::device> &Hidden_in, /* [1, 384, 8192] */
                   tensor<float, loc_t::device> &Output     /* [1, 384, 8192] */
) {
    thread_context ctx(bid, tid);
    tensor<float> hidden_in({1, 12, 8192});
    tensor<float> output({1, 12, 8192});
    tdma_load_async(hidden_in,
                    Hidden_in({0, 12 * (CORES * bid + tid), 0}, {1, 12, 8192}));
    tensor<float> sum({1, 12});
    tensor<float> sum_sqr({1, 12});
    reduce_sum_sqr(hidden_in, sum, sum_sqr);
    layernorm(hidden_in, sum, sum_sqr, output, 2, 8192, true);
    tdma_store_async(
        output, Output({0, 12 * (CORES * bid + tid), 0}, {1, 12, 8192}));
}
