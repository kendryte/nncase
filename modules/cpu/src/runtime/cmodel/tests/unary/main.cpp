#include "cluster_def.h"
#include <runtime_utils.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {                                             \
        block##b::thread##t::Unary_0(*buffer_0, *buffer_3);                    \
        return arg;                                                            \
    }

#define DEFINE_BFUNC(b)                                                        \
    DEFINE_TFUNC(b, 0)                                                         \
    DEFINE_TFUNC(b, 1)                                                         \
    DEFINE_TFUNC(b, 2)                                                         \
    DEFINE_TFUNC(b, 3)

static tensor<float, loc_t::device> *buffer_0;
static tensor<float, loc_t::device> *buffer_3;

DEFINE_BFUNC(0)
DEFINE_BFUNC(1)
DEFINE_BFUNC(2)
DEFINE_BFUNC(3)
DEFINE_BFUNC(4)
DEFINE_BFUNC(5)
DEFINE_BFUNC(6)
DEFINE_BFUNC(7)

void _start(hardware_context_mt *hw_ctx_impl, runtime_util_mt *rt_util_mt,
            nncase_mt_t *nncase_mt_impl, uint8_t **inputs) {
    global_hardware_init(hw_ctx_impl);
    runtime_util = *rt_util_mt;
    nncase_mt = *nncase_mt_impl;

    auto buffer_0_ = tensor<float, loc_t::device>(
        gsl::make_span((float *)inputs[0], 786432), {1, 384, 2048});
    buffer_0 = &buffer_0_;
    auto buffer_3_ = tensor<float, loc_t::device>(
        gsl::make_span((float *)inputs[1], 786432), {1, 384, 2048});
    buffer_3 = &buffer_3_;

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
}