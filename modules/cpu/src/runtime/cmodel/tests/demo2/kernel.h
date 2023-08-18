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

static bool w_loaded;
static tensor<float> wqh({8, 2048, 128});
static tensor<float> wkh({8, 2048, 128});
static tensor<float> wvh({8, 2048, 128}); // [8, 2048, 128]
static tensor<float> wm({1024, 8192});    // []
// static tensor<float> wfc1ih({1024, 8192});
// static tensor<float> wfc2ih({1024, 8192}); // [1024, 8192]
static tensor<float> yih({96, 1024});

// 8 head per block
// 2048 w-len per thread
void stage1_kernel(
    [[maybe_unused]] tensor<float, loc_t::device> &WQ, /* 64, 8192, 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &WK, /* 64, 8192, 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &WV, /* 64, 8192, 128 */
    [[maybe_unused]] tensor<float, loc_t::device> &WM, /* 8192, 8192 */
    [[maybe_unused]] tensor<float, loc_t::device> &QKH /* 64,384,384 */
) {
    thread_context ctx(bid, tid);
    // 1. Get xi
    auto xi = X({0, tid * 2048}, {384, 2048}); // [384, 2048]

    // 2. load wqh wkh wvh
    if (!w_loaded) {
        tdma_load_async(wqh, WQ({bid * 8, tid * 2048, 0}, {8, 2048, 128}), ctx);
        tdma_load_async(wkh, WK({bid * 8, tid * 2048, 0}, {8, 2048, 128}), ctx);
        tdma_load_async(wvh, WV({bid * 8, tid * 2048, 0}, {8, 2048, 128}), ctx);
        tdma_load_async(wm, WM({bid * 1024, 0}, {1024, 8192}), ctx);
        // tdma_load_async(wfc1ih, WFC1({bid * 1024, tid * 8192}, {1024,
        // 8192}));
        tdma_wait(ctx); // 等待 x, wqh, wkh, wvh ready
        w_loaded = true;
    }
    // 3. compute k, v
    set_sched_strategy(sched_strategy_t::pin_block_tensor);
    // 仅在 block 内调度线程的 tensor
    // 指令

    // [384, 2048] x [8, 2048, 64] = [8, 384, 128]
    tensor_block_mma_sync(xi, wkh, kh, false, ctx);
    // [384, 2048] x [8, 2048, 64] = [8, 384, 128]
    tensor_block_mma_sync(xi, wvh, vh, false, ctx);

    // 4. compute qk
    tensor<float> qih({8, 384, 128});
    // [384, 2048] x [8, 2048, 128] = [8, 384, 128]
    tensor_mma_sync(xi, wqh, qih);
    // [8, 384, 128] x [8, 128, 384] = [8, 384, 384]
    auto khT = view_transpose(kh, dims_t({0, 2, 1}));
    tensor_block_mma_sync(qih, khT, qkh, false, ctx);

    set_sched_strategy(sched_strategy_t::normal);
    if (tid == 0) {
        tdma_store_async(qkh, QKH({bid * 8, 0, 0}, {8, 384, 384}), ctx);
    }
    tdma_wait(ctx);
}

// 8 head per block
// 96 seq-len per thread
void stage2_kernel(
    [[maybe_unused]] tensor<float, loc_t::device> &Norm,   /* 8192, 8192 */
    [[maybe_unused]] tensor<float, loc_t::device> &Softmax, /* 64,384,384 */
    [[maybe_unused]] tensor<float, loc_t::device> &YM      /* 384,8192 */
) {
    thread_context ctx(bid, tid);
    // 5. compute softmax
    auto qkhi = qkh({0, tid * 96, 0}, {8, 96, 384});

    tensor<float> sih({8, 96, 384});
    softmax(qkhi, sih, 2);

    tdma_store_async(sih, Softmax({bid * 8, tid * 96, 0}, {8, 96, 384}), ctx);
    tdma_wait(ctx);

    // 6. compute y
    tensor<float> yihT({8, 96, 128});
    // [8, 96, 384] x [8, 384, 128] = [8, 96, 128]
    tensor_mma_sync(sih, vh, yihT);
    // 下面的 transpose 可以在计算 yih 的过程中通过指定输出 stride
    // 无代价地做到。
    /* [96, 8, 128] -> [96, 1024] */
    tensor<float> yih({96, 8, 128});
    transpose(yihT, yih, dims_t({1, 0, 2}));
    auto yihv = view(yih, dims_t({96, 1024}));

    tensor<float> ym({96, 8192});

    // [96,1024] * [1024,8192] -> [96, 8192]
    //  @t, @b       @b, @l        @t, @l
    // note 如添加上mm的, 此时wm的n不能在thread上切, 只能多算一部分,
    // 计算结束后倒是可以继续在thread切.
    tensor_mma_sync(yihv, wm, ym);
    tdma_all_reduce_async(ym, ym, reduce_op_t::sum, reduce_strategy_t::by_block,
                          ctx);
    if (bid == 0) {
        tdma_store_async(ym, YM({tid * 96, 0}, {96, 8192}), ctx);
    }

    // 7. Add and sum & sqr
    tensor<float> sum({96});
    tensor<float> sum_sqr({96});
    auto xi = X({tid * 96, bid * 1024}, {96, 1024}); // [96, 1024]
    std::function<float(float, float)> f = [](float a, float b) -> float {
        return a + b;
    };
    auto ym_b = ym({0, bid * 1024}, {96, 1024});
    __tensor_binary_sync(xi, ym_b, ym_b, f);
    // yih = add_and_sum_sqr(yih, xi, sum, sum_sqr);
    tensor_reduce_sum_sqr(ym_b, sum, sum_sqr);
    // __tdma_all_reduce_async(sum, sum_sqr); // All blocks & threads reduce
    tdma_all_reduce_async(sum, sum, reduce_op_t::sum,
                          reduce_strategy_t::by_block, ctx);
    tdma_all_reduce_async(sum_sqr, sum_sqr, reduce_op_t::sum,
                          reduce_strategy_t::by_block, ctx);
    tdma_wait(ctx);
    tensor_layernorm_sync(ym_b, sum, sum_sqr, 1, 8192);

    tdma_store_async(ym_b, Norm({tid * 96, bid * 1024}, {96, 1024}), ctx);
}
