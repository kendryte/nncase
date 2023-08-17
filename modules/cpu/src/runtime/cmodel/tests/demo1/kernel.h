// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

// tensor<float, loc_t::shared> X({384, 8192});
// tensor<float, loc_t::shared> kh({8, 384, 128}), vh({8, 384, 128}); //
// [8, 384, 128] tensor<float, loc_t::shared> qkh({8, 384, 384});    //
// [8, 384, 384] tensor<float, loc_t::shared> qkh({8, 384, 384});    //
// [8, 384, 384]
#include "thread_context.h"
using namespace shared;

static bool w_loaded = false;
// static tensor<float> wqh({8, 2048, 128});
static tensor<float> wkh({8, 2048, 128});
// static tensor<float> wvh({8, 2048, 128}); // [8, 2048, 128]
// static tensor<float> wfc1ih({1024, 2752});
// static tensor<float> wfc2ih({1024, 2752}); // [1024, 2752]
// static tensor<float> yih({96, 1024});

// 8 head per block
// 2048 w-len per thread
void stage1_kernel(
    // tensor<float, loc_t::device> &WQ,  /* 64, 8192, 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &WK,     /* 64, 8192, 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &K,      /* 64, 384, 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &Sum,    /* 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &RSum,   /* 384 */
    [[maybe_unused]] tensor<float, loc_t::device> &RSumSqr, /* 384 */
    [[maybe_unused]] tensor<float, loc_t::device> &Norm /* 384, 8192 */
    // [[maybe_unused]] tensor<float, loc_t::device> &Norm /* 384,8192 */
    //  tensor<float, loc_t::device> &WV,  /* 64, 8192, 128 */
    //  tensor<float, loc_t::device> &WFC1 /* 384, 8192*/
) {
    thread_context ctx(bid, tid);
    // 1. Get xi
    auto xi = X({0, tid * 2048}, {384, 2048}); // [384, 2048]

    // 2. load wqh wkh wvh
    if (!w_loaded) {
        // tdma_load_async(wqh, WQ({bid * 8, tid * 2048, 0}, {8, 2048, 128}));
        tdma_load_async(wkh, WK({bid * 8, tid * 2048, 0}, {8, 2048, 128}), ctx);
        // tdma_load_async(wvh, WV({bid * 8, tid * 2048, 0}, {8, 2048, 128}));
        // tdma_load_async(wfc1ih, WFC1({bid * 1024, tid * 8192}, {1024,
        // 8192})); tdma_wait(); // 等待 x, wqh, wkh, wvh ready
        w_loaded = true;
    }

    // 3. compute k, v
    // set_sched_strategy(
    //     sched_strategy_t::pin_block_tensor); // 仅在 block 内调度线程的
    //     tensor
    //                                          // 指令

    // tensor_block_mma_sync(xi, wkh, kh, false,
    //                       ctx); // [384, 2048] x [8, 2048, 64] = [8, 384,
    //                       128]

    // if (tid == 0) {
    //     tdma_store_async(kh, K({bid * 8, 0, 0}, {8, 384, 128}), ctx);
    // }
    // tdma_wait(ctx);

    // tensor<float> sum({128});
    // tensor_reduce_sync(wkh, sum, reduce_op_t::sum, 0.0f, dims_t({0, 1}),
    // false); tdma_all_reduce_async(sum, sum, reduce_op_t::sum, dims_t({0}),
    // ctx);

    // if (bid == 0 && tid == 0) {
    //     tdma_store_async(sum, std::move(Sum), ctx);
    // }

    // X [384,8192] => separate to bid , tid
    auto xj = X({0, (bid * CORES + tid) * 256}, {384, 256}); // [48,2048]
    tensor<float> r_sum({384});
    tensor<float> r_sum_sqr({384});
    tensor_reduce_sum_sqr(xj, r_sum, r_sum_sqr);
    tdma_all_reduce_async(r_sum, r_sum, reduce_op_t::sum, dims_t({0}), ctx);
    tdma_all_reduce_async(r_sum_sqr, r_sum_sqr, reduce_op_t::sum, dims_t({0}),
                          ctx);
    if (bid == 0 && tid == 0) {
        tdma_store_async(r_sum, std::move(RSum), ctx);
        tdma_store_async(r_sum_sqr, std::move(RSumSqr), ctx);
    }

    tensor<float> gamma({256});
    tdma_fill_async(gamma, 1.0f);
    tensor<float> beta({256});
    tdma_fill_async(beta, 1.0f);
    tensor_layernorm_sync(xj, r_sum, r_sum_sqr, gamma, beta, 1e-6f, 1, 8192);
    tdma_store_async(xj, Norm({0, (bid * CORES + tid) * 256}, {384, 256}),
                     ctx);

    // tensor_sum_sqr(xj, r_sum, r_sum_sqr);

    // tensor<float> layer_sum({8, 2048});
    // tensor<float> layer_sum_sqr({8, 2048});

    // vh = tensor_block_mma_sync(
    //     xi, wvh); // [384, 2048] x [8, 2048, 64] = [8, 384, 128]

    // // 4. compute qk
    // tensor<float> qih({8, 384, 128});
    // qih = tensor_mma_sync(
    //     xi, wqh); // [384, 2048] x [8, 2048, 128] = [8, 384, 128]
    // qkh = tensor_block_mma_sync(
    //     qih, kh.T); // [8, 384, 128] x [8, 128, 384] = [8, 384, 384]

    // set_sched_strategy(sched_strategy_t::normal);
}

// // 8 head per block
// // 96 seq-len per thread
// void stage2_kernel() {
//     // 5. compute softmax
//     auto qkhi = qkh({bid * 8, tid * 96, 0}, {8, 96, 384});

//     tensor<float, 8, 96, 384> sih;
//     sih = softmax(qkhi, 2);

//     // 6. compute y
//     tensor<float, 8, 96, 128> yihT;
//     yihT = __tensor_mma_sync(sih,
//                              vh); // [8, 96, 384] x [8, 384, 128] = [8, 96,
//                              128]
//     // 下面的 transpose 可以在计算 yih 的过程中通过指定输出 stride
//     // 无代价地做到。
//     yih = reshape(transpose(yihT, {1, 0, 2}) /* [96, 8, 128] */,
//                   {96, 1024}); // [96, 1024]

//     // 7. Add and sum & sqr
//     tensor<float, 96> sum, sum_sqr;
//     auto xi = X({tid * 96, bid * 1024}, {96, 1024}); // [96, 1024]
//     yih = add_and_sum_sqr(yih, xi, sum, sum_sqr);
//     __tdma_all_reduce_async(sum, sum_sqr,
//                             reduce_op_t::SUM); // All blocks & threads reduce
//     __tdma_wait();

//     // 8. compute LayerNorm
//     yih = layer_norm(yih, sum, sum_sqr); // [96, 1024]
// }

// //
// void stage3_kernel(tensor<float> &WFC1, tensor<float> &WFC2) {
//     // 9. fc1
//     tensor<float> fc1ih({96, 2752});
//     fc1ih =
//         __tensor_mma_sync(yih, fc1ih); // [96, 1024] x [1024, 8192] = [96,
//         8192]
// }
