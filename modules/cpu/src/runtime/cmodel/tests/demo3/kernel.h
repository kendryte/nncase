// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

#include "thread_context.h"
using namespace shared;

static bool w_loaded;
static tensor<float> v2_w({8, 2048, 128});
static tensor<float> v3_data({384, 128});
static tensor<int64_t> position_ids({1, 384});

void stage1_kernel(tensor<float, loc_t::device> &Hidden_in, /* [1, 384, 8192] */
                   tensor<float, loc_t::device> &V0_gamma,
                   tensor<float, loc_t::device> &V0_beta,
                   tensor<float, loc_t::device> &V2_w,    /* [64, 8192, 128] */
                   tensor<float, loc_t::device> &V3_data, /* [384, 128] */
                   tensor<int64_t, loc_t::device> &Position_ids /* [1,384] */
) {
    thread_context ctx(bid, tid);
    tensor<float> v0_gamma({2048});
    tensor<float> v0_beta({2048});
    tensor<float> v0({1, 48, 2048}); /* [1, 384, 8192] [1, 48@b, 2048@t]  */

    if (!w_loaded) {
        tdma_load_async(v0_gamma, V0_gamma({tid * 2048}, {2048}), ctx);
        tdma_load_async(v0_beta, V0_beta({tid * 2048}, {2048}), ctx);
        tdma_load_async(v2_w, V2_w({8 * bid, 2048 * tid, 0}, {8, 2048, 128}),
                        ctx);
        tdma_load_async(v3_data, std::move(V3_data), ctx);
        tdma_load_async(position_ids, std::move(Position_ids), ctx);

        tdma_wait(ctx);
    }

    tdma_load_async(v0, Hidden_in({0, bid * 48, tid * 2048}, {1, 48, 2048}),
                    ctx);
    {
        tensor<float> v0_sum({1, 48});
        tensor<float> v0_sum_sqr({1, 48});
        reduce_sum_sqr(v0, v0_sum, v0_sum_sqr);
        tdma_reduce_async(v0_sum, v0_sum, reduce_op_t::sum, ctx);
        tdma_reduce_async(v0_sum_sqr, v0_sum_sqr, reduce_op_t::sum, ctx);
        layernorm(v0, v0_sum, v0_sum_sqr, 2, 8192);
    } // v0 [1, 384, 8192] [1, 48@b, 8192]

    auto v1 = unsqueeze(v0); // [1, 1, 384, 8192] [1, 1, 48@b, 2048@t]
    /* 这里如果V2不shared的话可以只做thread间的reduce */
    tensor_block_mma_sync(v1, v2_w, V2, false, ctx);

    tensor<float> v3({1, 384, 128}); //
    gather(v3_data, position_ids, v3, 1);
}
