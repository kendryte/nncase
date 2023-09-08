#include "cluster_def.h"
#include <runtime_utils.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {                                             \
        block##b::thread##t::stage1_kernel(                                    \
            *Hidden_in, *V0_gamma, *V0_beta, *V2_w, *V16_w, *V31_w, *V35_w,    \
            *V3_data, *V11_data, *Attn_mask, *V38_w, *V40_w, *V42_w,           \
            *Position_ids, *Output, *V25, *GV31, ImmOutputs);                  \
        return arg;                                                            \
    }

#define DEFINE_BFUNC(b)                                                        \
    DEFINE_TFUNC(b, 0)                                                         \
    DEFINE_TFUNC(b, 1)                                                         \
    DEFINE_TFUNC(b, 2)                                                         \
    DEFINE_TFUNC(b, 3)

static tensor<float, loc_t::device> *Hidden_in;      // ({1, 384, 8192})
static tensor<float, loc_t::device> *V0_gamma;       // ({8192})
static tensor<float, loc_t::device> *V0_beta;        // ({8192})
static tensor<float, loc_t::device> *V2_w;           // ({64, 8192, 128})
static tensor<float, loc_t::device> *V16_w;          // ({64, 8192, 128})
static tensor<float, loc_t::device> *V31_w;          // ({64, 8192, 128})
static tensor<float, loc_t::device> *V35_w;          // ({8192, 8192})
static tensor<float, loc_t::device> *V3_data;        // ({384, 128})
static tensor<float, loc_t::device> *V11_data;       // ({384, 128})
static tensor<float, loc_t::device> *Attn_mask;      // ({1, 1, 384, 384})
static tensor<float, loc_t::device> *V38_w;          // ({8192, 22016})
static tensor<float, loc_t::device> *V40_w;          // ({8192, 22016})
static tensor<float, loc_t::device> *V42_w;          // ({22016, 8192})
static tensor<int64_t, loc_t::device> *Position_ids; // ({1, 384})
static tensor<float, loc_t::device> *Output;         // ({1, 384, 8192})
static tensor<float, loc_t::device> *V25;            // ({1, 64, 128, 384})
static tensor<float, loc_t::device> *GV31;           // ({1, 64, 384, 128})

constexpr int OutNum = 11;
// static tensor<float, loc_t::device> ImmOutputs[OutNum] = {
//     tensor<float, loc_t::device>({1, 384, 8192}),
//     tensor<float, loc_t::device>({1, 64, 384, 128}),
//     tensor<float, loc_t::device>({1, 64, 384, 128}),
//     tensor<float, loc_t::device>({1, 64, 384, 128}),
//     tensor<float, loc_t::device>({1, 384, 128}),
//     tensor<float, loc_t::device>({1, 64, 384, 128}),
//     tensor<float, loc_t::device>({1, 64, 384, 384}),
//     tensor<float, loc_t::device>({1, 64, 384, 128}),
//     tensor<float, loc_t::device>({1, 384, 8192}),
//     tensor<float, loc_t::device>({1, 384, 22016}),
//     tensor<float, loc_t::device>({1, 384, 8192}),
// };
static tensor<float, loc_t::device> *ImmOutputs[OutNum];

DEFINE_BFUNC(0)
DEFINE_BFUNC(1)
DEFINE_BFUNC(2)
DEFINE_BFUNC(3)
DEFINE_BFUNC(4)
DEFINE_BFUNC(5)
DEFINE_BFUNC(6)
DEFINE_BFUNC(7)

#define MALLOC_GLOBAL(i, name, type, size, shape)                              \
    auto g_##name = tensor<type, loc_t::device>(                               \
        gsl::make_span((type *)inputs[i], size), shape);                       \
    name = &g_##name;

#define MALLOC_IMM(i, name, type, size, shape)                                 \
    auto g_##i = tensor<type, loc_t::device>(                                  \
        gsl::make_span((type *)inputs[i], size), shape);                       \
    name = &g_##i;

#define MALLOC_SHARED(name, b, type, size, shape)                              \
    auto shared##b##name = tensor<type, loc_t::shared>(shape);                 \
    block##b::shared::name = &shared##b##name;

#define CLEAR_SHARED(i)                                                        \
    tdma_fill_async<float>(*block##i::shared::V2, 0);                          \
    tdma_fill_async<float>(*block##i::shared::V5, 0);                          \
    tdma_fill_async<float>(*block##i::shared::V13, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V16, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V25, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V26, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V31, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V32, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V33, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V35, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V38, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V40, 0);                         \
    tdma_fill_async<float>(*block##i::shared::V42, 0);

void _start(hardware_context_mt *hw_ctx_impl, runtime_util_mt *rt_util_mt,
            nncase_mt_t *nncase_mt_impl, uint8_t **inputs) {
    global_hardware_init(hw_ctx_impl);
    runtime_util = *rt_util_mt;
    nncase_mt = *nncase_mt_impl;

    MALLOC_IMM(15, ImmOutputs[0], float, (1 * 384 * 8192),
               dims_t({1, 384, 8192}))
    MALLOC_IMM(16, ImmOutputs[1], float, (1 * 64 * 384 * 128),
               dims_t({1, 64, 384, 128}))
    MALLOC_IMM(17, ImmOutputs[2], float, (1 * 64 * 384 * 128),
               dims_t({1, 64, 384, 128}))
    MALLOC_IMM(18, ImmOutputs[3], float, (1 * 64 * 384 * 128),
               dims_t({1, 64, 384, 128}))
    MALLOC_IMM(19, ImmOutputs[4], float, (1 * 384 * 128), dims_t({1, 384, 128}))
    MALLOC_IMM(20, ImmOutputs[5], float, (1 * 64 * 384 * 128),
               dims_t({1, 64, 384, 128}))
    MALLOC_IMM(21, ImmOutputs[6], float, (1 * 64 * 384 * 384),
               dims_t({1, 64, 384, 384}))
    // v32
    MALLOC_IMM(22, ImmOutputs[7], float, (1 * 64 * 384 * 128),
               dims_t({1, 64, 384, 128}))
    // v35
    MALLOC_IMM(23, ImmOutputs[8], float, (1 * 384 * 8192),
               dims_t({1, 384, 8192}))
    // v38
    MALLOC_IMM(24, ImmOutputs[9], float, (1 * 384 * 22016),
               dims_t({1, 384, 22016}))

    // v43
    MALLOC_IMM(25, ImmOutputs[10], float, (1 * 384 * 8192),
               dims_t({1, 384, 8192}))

    MALLOC_GLOBAL(0, Hidden_in, float, (1 * 384 * 8192), dims_t({1, 384, 8192}))
    MALLOC_GLOBAL(1, V0_gamma, float, (8192), dims_t({8192}))
    MALLOC_GLOBAL(2, V0_beta, float, (8192), dims_t({8192}))
    MALLOC_GLOBAL(3, V2_w, float, (64 * 8192 * 128), dims_t({64, 8192, 128}))
    MALLOC_GLOBAL(4, V16_w, float, (64 * 8192 * 128), dims_t({64, 8192, 128}))
    MALLOC_GLOBAL(5, V31_w, float, (64 * 8192 * 128), dims_t({64, 8192, 128}))
    MALLOC_GLOBAL(6, V35_w, float, (8192 * 8192), dims_t({8192, 8192}))
    MALLOC_GLOBAL(7, V3_data, float, (384 * 128), dims_t({384, 128}))
    MALLOC_GLOBAL(8, V11_data, float, (384 * 128), dims_t({384, 128}))
    MALLOC_GLOBAL(9, Attn_mask, float, (1 * 1 * 384 * 384),
                  dims_t({1, 1, 384, 384}))
    MALLOC_GLOBAL(10, V38_w, float, (8192 * 22016), dims_t({8192, 22016}))
    MALLOC_GLOBAL(11, V40_w, float, (8192 * 22016), dims_t({8192, 22016}))
    MALLOC_GLOBAL(12, V42_w, float, (22016 * 8192), dims_t({22016, 8192}))
    MALLOC_GLOBAL(13, Position_ids, int64_t, (1 * 384), dims_t({1, 384}))
    MALLOC_GLOBAL(14, Output, float, (1 * 384 * 8192), dims_t({1, 384, 8192}))
    auto V25_tmp = tensor<float, loc_t::device>(dims_t({1, 64, 128, 384}));
    auto GV31_tmp = tensor<float, loc_t::device>(dims_t({1, 64, 384, 128}));
    V25 = &V25_tmp;
    GV31 = &GV31_tmp;

    MALLOC_SHARED(V2, 0, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 1, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 2, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 3, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 4, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 5, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 6, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V2, 7, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))

    MALLOC_SHARED(V5, 0, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 1, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 2, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 3, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 4, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 5, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 6, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V5, 7, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))

    MALLOC_SHARED(V13, 0, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 1, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 2, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 3, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 4, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 5, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 6, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V13, 7, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))

    MALLOC_SHARED(V16, 0, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 1, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 2, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 3, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 4, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 5, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 6, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V16, 7, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))

    MALLOC_SHARED(V25, 0, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 1, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 2, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 3, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 4, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 5, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 6, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))
    MALLOC_SHARED(V25, 7, float, (1 * 64 * 128 * 384),
                  dims_t({1, 64, 128, 384}))

    MALLOC_SHARED(V26, 0, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 1, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 2, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 3, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 4, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 5, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 6, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))
    MALLOC_SHARED(V26, 7, float, (1 * 64 * 48 * 384), dims_t({1, 64, 48, 384}))

    MALLOC_SHARED(V31, 0, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 1, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 2, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 3, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 4, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 5, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 6, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V31, 7, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))

    MALLOC_SHARED(V32, 0, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 1, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 2, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 3, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 4, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 5, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 6, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))
    MALLOC_SHARED(V32, 7, float, (1 * 64 * 48 * 128), dims_t({1, 64, 48, 128}))

    MALLOC_SHARED(V33, 0, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 1, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 2, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 3, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 4, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 5, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 6, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))
    MALLOC_SHARED(V33, 7, float, (1 * 48 * 64 * 128), dims_t({1, 48, 64, 128}))

    MALLOC_SHARED(V35, 0, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 1, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 2, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 3, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 4, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 5, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 6, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V35, 7, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))

    MALLOC_SHARED(V38, 0, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 1, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 2, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 3, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 4, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 5, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 6, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V38, 7, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))

    MALLOC_SHARED(V40, 0, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 1, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 2, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 3, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 4, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 5, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 6, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))
    MALLOC_SHARED(V40, 7, float, (1 * 48 * 22016), dims_t({1, 48, 22016}))

    MALLOC_SHARED(V42, 0, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 1, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 2, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 3, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 4, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 5, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 6, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))
    MALLOC_SHARED(V42, 7, float, (1 * 48 * 8192), dims_t({1, 48, 8192}))

    pthread_t t_0_0, t_1_0, t_2_0, t_3_0, t_4_0, t_5_0, t_6_0, t_7_0;
    pthread_t t_0_1, t_1_1, t_2_1, t_3_1, t_4_1, t_5_1, t_6_1, t_7_1;
    pthread_t t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2, t_6_2, t_7_2;
    pthread_t t_0_3, t_1_3, t_2_3, t_3_3, t_4_3, t_5_3, t_6_3, t_7_3;

    runtime_util.create_thread(t_0_0, NULL, f_0_0);
    runtime_util.create_thread(t_0_1, NULL, f_0_1);
    runtime_util.create_thread(t_0_2, NULL, f_0_2);
    runtime_util.create_thread(t_0_3, NULL, f_0_3);
    runtime_util.create_thread(t_1_0, NULL, f_1_0);
    runtime_util.create_thread(t_1_1, NULL, f_1_1);
    runtime_util.create_thread(t_1_2, NULL, f_1_2);
    runtime_util.create_thread(t_1_3, NULL, f_1_3);
    runtime_util.create_thread(t_2_0, NULL, f_2_0);
    runtime_util.create_thread(t_2_1, NULL, f_2_1);
    runtime_util.create_thread(t_2_2, NULL, f_2_2);
    runtime_util.create_thread(t_2_3, NULL, f_2_3);
    runtime_util.create_thread(t_3_0, NULL, f_3_0);
    runtime_util.create_thread(t_3_1, NULL, f_3_1);
    runtime_util.create_thread(t_3_2, NULL, f_3_2);
    runtime_util.create_thread(t_3_3, NULL, f_3_3);
    runtime_util.create_thread(t_4_0, NULL, f_4_0);
    runtime_util.create_thread(t_4_1, NULL, f_4_1);
    runtime_util.create_thread(t_4_2, NULL, f_4_2);
    runtime_util.create_thread(t_4_3, NULL, f_4_3);
    runtime_util.create_thread(t_5_0, NULL, f_5_0);
    runtime_util.create_thread(t_5_1, NULL, f_5_1);
    runtime_util.create_thread(t_5_2, NULL, f_5_2);
    runtime_util.create_thread(t_5_3, NULL, f_5_3);
    runtime_util.create_thread(t_6_0, NULL, f_6_0);
    runtime_util.create_thread(t_6_1, NULL, f_6_1);
    runtime_util.create_thread(t_6_2, NULL, f_6_2);
    runtime_util.create_thread(t_6_3, NULL, f_6_3);
    runtime_util.create_thread(t_7_0, NULL, f_7_0);
    runtime_util.create_thread(t_7_1, NULL, f_7_1);
    runtime_util.create_thread(t_7_2, NULL, f_7_2);
    runtime_util.create_thread(t_7_3, NULL, f_7_3);

    runtime_util.join_thread(t_0_0);
    runtime_util.join_thread(t_0_1);
    runtime_util.join_thread(t_0_2);
    runtime_util.join_thread(t_0_3);
    runtime_util.join_thread(t_1_0);
    runtime_util.join_thread(t_1_1);
    runtime_util.join_thread(t_1_2);
    runtime_util.join_thread(t_1_3);
    runtime_util.join_thread(t_2_0);
    runtime_util.join_thread(t_2_1);
    runtime_util.join_thread(t_2_2);
    runtime_util.join_thread(t_2_3);
    runtime_util.join_thread(t_3_0);
    runtime_util.join_thread(t_3_1);
    runtime_util.join_thread(t_3_2);
    runtime_util.join_thread(t_3_3);
    runtime_util.join_thread(t_4_0);
    runtime_util.join_thread(t_4_1);
    runtime_util.join_thread(t_4_2);
    runtime_util.join_thread(t_4_3);
    runtime_util.join_thread(t_5_0);
    runtime_util.join_thread(t_5_1);
    runtime_util.join_thread(t_5_2);
    runtime_util.join_thread(t_5_3);
    runtime_util.join_thread(t_6_0);
    runtime_util.join_thread(t_6_1);
    runtime_util.join_thread(t_6_2);
    runtime_util.join_thread(t_6_3);
    runtime_util.join_thread(t_7_0);
    runtime_util.join_thread(t_7_1);
    runtime_util.join_thread(t_7_2);
    runtime_util.join_thread(t_7_3);

    CLEAR_SHARED(0);
    CLEAR_SHARED(1);
    CLEAR_SHARED(2);
    CLEAR_SHARED(3);
    CLEAR_SHARED(4);
    CLEAR_SHARED(5);
    CLEAR_SHARED(6);
    CLEAR_SHARED(7);
}