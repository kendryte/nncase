// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

#include "thread_context.h"
using namespace shared;

static bool w_loaded;
static tensor<float> v2_w({64, 2048, 128});
static tensor<float> v16_w({64, 2048, 128});
static tensor<float> v31_w({64, 2048, 128});
static tensor<float> v35_w({2048, 8192});
static tensor<float> v3_data({384, 32});
static tensor<float> v11_data({384, 32});
static tensor<int64_t> position_ids({1, 48});
static tensor<float> v38_w({2048, 22016});
static tensor<float> v40_w({2048, 22016});
static tensor<float> v42_w({5504, 8192});
#define DUMP 0

void stage1_kernel(
    tensor<float, loc_t::device> &Hidden_in,               /* [1, 384, 8192] */
    tensor<float, loc_t::device> &V0_gamma,                /* [8192] */
    tensor<float, loc_t::device> &V0_beta,                 /* [8192] */
    tensor<float, loc_t::device> &V2_w,                    /* [64, 8192, 128] */
    tensor<float, loc_t::device> &V16_w,                   /* [64, 8192, 128] */
    tensor<float, loc_t::device> &V31_w,                   /* [64, 8192, 128] */
    tensor<float, loc_t::device> &V35_w,                   /* [8192, 8192] */
    tensor<float, loc_t::device> &V3_data,                 /* [384, 128] */
    tensor<float, loc_t::device> &V11_data,                /* [384, 128] */
    tensor<float, loc_t::device> &Attn_mask,               /* [1,1,384,384] */
    tensor<float, loc_t::device> &V38_w,                   /* [8192, 22016] */
    tensor<float, loc_t::device> &V40_w,                   /* [8192, 22016] */
    tensor<float, loc_t::device> &V42_w,                   /* [22016, 8192] */
    tensor<int64_t, loc_t::device> &Position_ids,          /* [1,384] */
    [[maybe_unused]] tensor<float, loc_t::device> &Output, /* [1,384,8192] */
    tensor<float, loc_t::device> &V25, tensor<float, loc_t::device> &GV31,
    [[maybe_unused]] tensor<float, loc_t::device> *ImmOutputs) {
    thread_context ctx(bid, tid);
    tensor<float> v0_gamma({2048});
    tensor<float> v0_beta({2048});
    tensor<float> v0({1, 48, 2048});    /* [1, 384, 8192] [1, 48@b, 2048@t]  */
    tensor<float> var_5({1, 48, 2048}); /* [1, 384, 8192] [1, 48@b, 2048@t]  */

    tensor<float> attn_mask({1, 1, 48, 384});

    if (!w_loaded) {
        tdma_load_async(v0_gamma, V0_gamma({tid * 2048}, {2048}), ctx);
        tdma_load_async(v0_beta, V0_beta({tid * 2048}, {2048}), ctx);
        tdma_load_async(v2_w, V2_w({0, 2048 * tid, 0}, {64, 2048, 128}), ctx);
        tdma_load_async(v16_w, V16_w({0, 2048 * tid, 0}, {64, 2048, 128}), ctx);
        tdma_load_async(v31_w, V31_w({0, 2048 * tid, 0}, {64, 2048, 128}), ctx);
        tdma_load_async(v35_w, V35_w({2048 * tid, 0}, {2048, 8192}), ctx);
        tdma_load_async(v38_w, V38_w({2048 * tid, 0}, {2048, 22016}), ctx);
        tdma_load_async(v40_w, V40_w({2048 * tid, 0}, {2048, 22016}), ctx);
        tdma_load_async(v42_w, V42_w({5504 * tid, 0}, {5504, 8192}), ctx);
        tdma_load_async(v3_data, V3_data({0, 32 * tid}, {384, 32}), ctx);
        tdma_load_async(v11_data, V11_data({0, 32 * tid}, {384, 32}), ctx);
        tdma_load_async(position_ids, Position_ids({0, 48 * bid}, {1, 48}),
                        ctx);
        tdma_load_async(attn_mask,
                        Attn_mask({0, 0, 48 * bid, 0}, {1, 1, 48, 384}), ctx);

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
        layernorm(v0, v0_sum, v0_sum_sqr, v0, 2, 8192, true);
#if DUMP
        {
            tdma_store_async(
                v0, ImmOutputs[0]({0, 48 * bid, 2048 * tid}, {1, 48, 2048}),
                ctx);
        }
#endif
    } // v0 [1, 384, 8192] [1, 48@b, 8192]

    auto v1 = unsqueeze(v0); // [1, 1, 384, 8192] [1, 1, 48@b, 2048@t]
    /*
      [1, 1, 384, 8192]    @ [64, 8192, 128]  -> [1, 64, 384, 128]
      [1, 1, 48@b, 2048@t] @ [64, 2048@t, 128] -> [1, 64, 48@b, 128] @ shared
     */
    tensor_block_mma_sync(v1, v2_w, V2, false, ctx);
#if DUMP
    if (tid == 0) {
        tdma_store_async(
            V2, ImmOutputs[1]({0, 0, 48 * bid, 0}, {1, 64, 48, 128}), ctx);
    }
    tdma_all_wait(ctx);
#endif

    /*
      [384, 128][[1, 384]] -> [1, 384, 128]
      [384, 32@t][[1, 48@b]] -> [1, 48@b, 32@t]
    */
    tensor<float> v3({1, 48, 32});
    gather(v3_data, position_ids, v3, 0);
#if DUMP
    {
        tdma_store_async(
            v3, ImmOutputs[4]({0, 48 * bid, 32 * tid}, {1, 48, 32}), ctx);
    }
#endif
    auto v4 = unsqueeze(v3); /* 1, 1, 48, 32 */

    /*
      [1, 64, 384, 128]           X   [1, 1, 384, 128] ->  [1, 64, 384, 128]
      [1, 64, 48@b, 32@t] @ shared X   [1, 1, 48@b, 32@t] ->  [1, 64, 48@b,
      32@t] @ shared
    */
    auto v5_ = V5({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    auto v2_ = V2({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    binary(v2_, v4, v5_, binary_op_t::mul);

    /*
      [1, 64, 384, 64:] -> [1, 64, 384, 64:]
      [1, 64, 48@b, 16@t]@shared ->  [1, 16@t, 48@b, 16@t]
    */
    auto v6 = V2({0, 0, 0, 64 + 16 * tid}, {1, 64, 48, 16});
    tensor<float> v7({1, 64, 48, 16});
    tdma_wait(ctx);
    unary(v6, v7, unary_op_t::neg);

    __tensor_copy_sync(V2({0, 0, 0, 64 + 16 * tid}, {1, 64, 48, 16}),
                       V2({0, 0, 0, 16 * tid}, {1, 64, 48, 16}));
    __tensor_copy_sync(V2({0, 0, 0, 16 * tid}, {1, 64, 48, 16}), std::move(v7));
    tdma_wait(ctx);

    /* v10 为concat(v7,v8), 实际来源于V2，因此无需实际动作 */
    auto v10 = V2({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
#if DUMP
    {
        tdma_store_async(
            v10, ImmOutputs[2]({0, 0, 48 * bid, 32 * tid}, {1, 64, 48, 32}),
            ctx);
    }
#endif

    /*
      [384, 128][[1, 384]] -> [1, 384, 128]
      [384, 32@t][[1, 48@b]] -> [1, 48@b, 32@t]
    */
    tensor<float> v11({1, 48, 32});
    gather(v11_data, position_ids, v11, 0);
    auto v12 = unsqueeze(v11); // [1, 1, 384, 128] [1, 1, 48@b, 32@t]

    /*
      [1, 64, 384, 128]           X   [1, 1, 384, 128] ->  [1, 64, 384, 128]
      [1, 64, 48@b, 32@t] @ shared X   [1, 1, 48@b, 32@t] ->  [1, 64, 48@b,
      32@t] @ shared
    */
    auto v13_ = V13({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    // auto v2_ = V2({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    binary(v2_, v12, v13_, binary_op_t::mul);

    tdma_all_wait(ctx); /* wait v5,v13 */

    /*
    [1, 64, 384, 128] + [1, 64, 384, 128] -> [1, 64, 384, 128]
    [1, 64, 48@b, 32@t] + [1, 64, 48@b, 32@t] -> [1, 64, 48@b, 32@t]
    */
    tensor<float> v14({1, 64, 48, 32});
    auto v5__ = V5({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    auto v13__ = V13({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    binary(v5__, v13__, v14, binary_op_t::add);
#if DUMP
    {
        tdma_store_async(
            v14, ImmOutputs[3]({0, 0, 48 * bid, 32 * tid}, {1, 64, 48, 32}),
            ctx);
    }
#endif
    tensor<float> &v15 = v0;
    /*
      [1,1,384,8192] @ [64, 8192, 128] -> [1,64,384,128]
      [1, 1, 48@b, 2048@t] @ [64, 2048@t, 128] -> [1,64,48@b,128]@shared
    */
    tensor_block_mma_sync(v15, v16_w, V16, false, ctx);

    /*
      [1,64,384,128] X [1,64,384,128] -> [1,64,384,128]
      [1,64,48@b,32@t]@shared X [1,1,48@b,32@t] ->  [1,64,48@b,32@t]
    */
    tensor<float> v17({1, 64, 48, 32});
    auto v16 = V16({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    binary(v16, v4, v17, binary_op_t ::mul);

    /*
      [1, 64, 384, 64:] -> [1, 64, 384, 64:]
      [1, 64, 48@b, 64]@shared ->  [1, 16@t, 48@b, 64]@shared
    */
    tensor<float> v19({1, 64, 48, 16});
    auto v18 = V16({0, 0, 0, 64 + 16 * tid}, {1, 64, 48, 16});
    tdma_wait(ctx);
    unary(v18, v19, unary_op_t::neg);
    __tensor_copy_sync(V16({0, 0, 0, 64 + 16 * tid}, {1, 64, 48, 16}),
                       V16({0, 0, 0, 16 * tid}, {1, 64, 48, 16}));
    __tensor_copy_sync(V16({0, 0, 0, 16 * tid}, {1, 64, 48, 16}),
                       std::move(v19));

    /*
      [1,64,384,128] x [1,1,384,128] ->  [1,64,384,128]
      [1, 64, 48@b, 32@t]@shared x [1, 1, 48@b, 32@t] -> [1, 64, 48@b, 32@t]
    */
    tensor<float> v23({1, 64, 48, 32});
    auto v22 = V16({0, 0, 0, 32 * tid}, {1, 64, 48, 32});
    tdma_wait(ctx);
    binary(v22, v12, v23, binary_op_t::mul);
#if DUMP
    {
        tdma_store_async(
            v22, ImmOutputs[5]({0, 0, 48 * bid, 32 * tid}, {1, 64, 48, 32}),
            ctx);
    }
#endif

    tensor<float> v24({1, 64, 48, 32});
    /*
      [1,64,384,128] + [1,1,384,128] ->  [1,64,384,128]
      [1, 64, 48@b, 32@t] + [1, 1, 48@b, 32@t] -> [1, 64, 48@b, 32@t]
    */
    binary(v17, v23, v24, binary_op_t::add);

    tensor<float> v25({1, 64, 32, 48});
    transpose(v24, v25, dims_t({0, 1, 3, 2}));

    /*
      [1, 64, 384, 128] @ [1, 64, 128, 384] -> [1, 64, 384, 384]
      [1, 64, 48@b, 32@t] @ [1, 64, 32@t, 384] -> [1, 64, 48@b, 384]@shared
    */
    tdma_store_async(v25, V25({0, 0, 32 * tid, 48 * bid}, {1, 64, 32, 48}),
                     ctx);
    tdma_all_wait(ctx); /* for store */
    tensor<float> v25_1({1, 64, 32, 384});
    tdma_load_async(v25_1, V25({0, 0, 32 * tid, 0}, {1, 64, 32, 384}), ctx);
    tdma_all_wait(ctx);
    tensor_block_mma_sync(v14, v25_1, V26, false, ctx);

    /* [1, 64, 384, 384]           / 11.31370  -> [1, 64, 384, 384]
       [1, 16@t, 48@b, 384]@shared   / 11.31370 -> [1, 16@t, 48@b, 384]
    */
    auto v26 = V26({0, 16 * tid, 0, 0}, {1, 16, 48, 384});
    tensor<float> v27({1, 16, 48, 384});
    auto v26_c = tensor<float>({1, 1, 1, 1});
    tdma_fill_async(v26_c, 11.313708f);
    binary(v26, v26_c, v27, binary_op_t::div);

    /* [1, 64, 384, 384] + [1,1,384,384] -> [1, 64, 384, 384]
       [1, 16@t, 48@b, 384] + [1, 1, 48@b, 384] -> [1, 16@t, 48@b,
       384]
    */
    auto &v28 = v27;
    binary(v27, attn_mask, v28, binary_op_t::add);
#if DUMP
    {
        tdma_store_async(
            v28, ImmOutputs[6]({0, 16 * tid, 48 * bid, 0}, {1, 16, 48, 384}),
            ctx);
    }
#endif
    /*
      [1, 64, 384, 384] -> [1, 64, 384, 384]
      [1, 16@t, 48@b, 384]@shared -> [1, 16@t, 48@b, 384]
    */
    tensor<float> v29({1, 16, 48, 384}); //[1, 64, 384, 384] [1, 8@b, 96@t, 384]
    softmax(v28, v29, 3);

    /*
      [1,1,384,8192] @ [64, 8192, 128] -> [1,64,384,128]
      [1, 1, 48@b, 2048@t] @ [64, 2048@t, 128] -> [1, 64, 48@b, 128]@shared
    */
    auto &v30 = v1;
    tensor_block_mma_sync(v30, v31_w, V31, false, ctx);

    /* need resplit V */
    auto v31 = V31({0, 16 * tid, 0, 0}, {1, 16, 48, 128});
    tdma_store_async(v31, GV31({0, 16 * tid, 48 * bid, 0}, {1, 16, 48, 128}),
                     ctx);

    /*
      [1,64,384,384] @ [1,64,384,128] ->  [1,64,384,128]
      [1, 16@t, 48@b, 384] @ [1, 16@t, 384, 128] -> [1, 16@t, 48@b, 128]@shared
    */
    tdma_all_wait(ctx);
    tensor<float> v31_resplit({1, 16, 384, 128});
    tdma_load_async(v31_resplit, GV31({0, 16 * tid, 0, 0}, {1, 16, 384, 128}),
                    ctx);
    auto v32 = V32({0, 16 * tid, 0, 0}, {1, 16, 48, 128});
    matmul(v29, v31_resplit, v32);
    tdma_wait(ctx);
#if DUMP
    {
        tdma_store_async(
            v32, ImmOutputs[7]({0, 16 * tid, 48 * bid, 0}, {1, 16, 48, 128}),
            ctx);
    }
#endif
    /*
      [1, 64, 384, 128] -> [1, 384, 64, 128]
      [1, 64, 48@b, 128]@shared -> [1, 48@b, 64, 128]@shared
     */
    auto v33 = V33({0, 0, 16 * tid, 0}, {1, 48, 16, 128});
    transpose(v32, v33, {0, 2, 1, 3});

    /*
    [1, 384, 8192] @ [8192, 8192] -> [1,384,8192]
    [1, 48@b, 2048@t]@shared @ [2048@t, 8192] -> [1, 48@b, 8192]@shared
     */
    auto V34 = view(V33, {1, 48, 8192});
    auto v34 = V34({0, 0, 2048 * tid}, {1, 48, 2048});
    tdma_wait(ctx);
    tensor_block_mma_sync(v34, v35_w, V35, false, ctx);
#if DUMP
    {
        auto v35 = V35({0, 0, tid * 2048}, {1, 48, 2048});
        tdma_store_async(
            v35, ImmOutputs[8]({0, 48 * bid, 2048 * tid}, {1, 48, 2048}), ctx);
    }
#endif

    /*
    [1,384,8192] +  [1,384,8192]  -> [1,384,8192]
    [1, 48@b, 2048@tid]@shared +  [1, 48@b, 2048@tid] -> [1, 48@b, 2048@tid]
     */
    tdma_load_async(var_5, Hidden_in({0, bid * 48, tid * 2048}, {1, 48, 2048}),
                    ctx);
    auto v35 = V35({0, 0, tid * 2048}, {1, 48, 2048});
    tensor<float> v36({1, 48, 2048}); /* [1, 384, 8192] [1, 48@b, 2048@t] */
    binary(var_5, v35, v36, binary_op_t::add);

    tensor<float> v37({1, 48, 2048}); /* [1, 384, 8192] [1, 48@b, 2048@t] */
    {
        tensor<float> v36_sum({1, 48});
        tensor<float> v36_sum_sqr({1, 48});
        reduce_sum_sqr(v36, v36_sum, v36_sum_sqr);
        tdma_reduce_async(v36_sum, v36_sum, reduce_op_t::sum, ctx);
        tdma_reduce_async(v36_sum_sqr, v36_sum_sqr, reduce_op_t::sum, ctx);
        layernorm(v36, v36_sum, v36_sum_sqr, v37, 2, 8192, true);
    }
    /*
      [1,384,8192] @ [8192, 22016] -> [1,384, 22016]
      [1, 48@b, 2048@t] @ [2048@t, 22016] -> [1, 48@b, 22016]@shared
    */
    tensor_block_mma_sync(v37, v38_w, V38, false, ctx);
#if DUMP
    {
        auto v38 = V38({0, 0, 5504 * tid}, {1, 48, 5504});
        tdma_store_async(
            v38, ImmOutputs[9]({0, 48 * bid, 5504 * tid}, {1, 48, 5504}), ctx);
    }
#endif

    /*
      [1,384, 22016] -> [1,384, 22016]
      [1, 48@b, 5504@t]@shared -> [1, 48@b, 5504@t]@shared
    */
    auto v39 = V38({0, 0, 5504 * tid}, {1, 48, 5504});
    unary(v39, v39, unary_op_t::swish);

    /* [1, 384, 8192]    @ [8192, 22016] -> [1, 384, 22016]
       [1, 48@b, 2048@t] @  [2048@t, 22016] -> [1, 48@b, 22016] @shared
    */
    tensor_block_mma_sync(v37, v40_w, V40, false, ctx);
    /*
    [1, 384, 22016] X  [1, 384, 22016] -> [1, 384, 22016]
    [1, 48@b, 5504@t]@shared X [1,48@b,5504@t]@shared -> [1, 48@b, 5504@t]@local
    */
    tensor<float> v41({1, 48, 5504});
    auto v40 = V40({0, 0, 5504 * tid}, {1, 48, 5504});
    binary(v39, v40, v41, binary_op_t::mul);

    /*
      [1, 384, 22016] @ [22016, 8192] -> [1, 384, 8192]
      [1, 48@b, 5504@t] @ [5504@t, 8192] -> [1, 48@b, 8192] @shared
    */
    tensor_block_mma_sync(v41, v42_w, V42, false, ctx);

    /*
      [1, 384, 8192] + [1, 384, 8192] -> [1, 384, 8192]
      [1, 48@b, 2048@t] + [1, 48@b, 2048@t] @shared -> [1, 48@b, 2048@t]
    */
    tensor<float> v43({1, 48, 2048});
    auto v42 = V42({0, 0, 2048 * tid}, {1, 48, 2048});
    binary(v36, v42, v43, binary_op_t::add);
    tdma_store_async(v43, Output({0, 48 * bid, 2048 * tid}, {1, 48, 2048}),
                     ctx);
#if DUMP
    {
        tdma_store_async(
            v43, ImmOutputs[10]({0, 48 * bid, 2048 * tid}, {1, 48, 2048}), ctx);
    }
#endif
}
