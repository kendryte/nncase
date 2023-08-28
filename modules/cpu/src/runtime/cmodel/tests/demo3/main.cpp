#include "cluster_def.h"
#include <io_utils.h>
#include <runtime_utils.h>
// #include <pthread.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {                                             \
        block##b::thread##t::stage1_kernel(                                    \
            Hidden_in, V0_gamma, V0_beta, V2_w, V16_w, V31_w, V35_w, V3_data,  \
            V11_data, Attn_mask, V38_w, V40_w, V42_w, Position_ids, Output);   \
        return arg;                                                            \
    }

#define DEFINE_BFUNC(b)                                                        \
    DEFINE_TFUNC(b, 0)                                                         \
    DEFINE_TFUNC(b, 1)                                                         \
    DEFINE_TFUNC(b, 2)                                                         \
    DEFINE_TFUNC(b, 3)

static tensor<float, loc_t::device> Hidden_in({1, 384, 8192});
static tensor<float, loc_t::device> V0_gamma({8192});
static tensor<float, loc_t::device> V0_beta({8192});
static tensor<float, loc_t::device> V2_w({64, 8192, 128});
static tensor<float, loc_t::device> V16_w({64, 8192, 128});
static tensor<float, loc_t::device> V31_w({64, 8192, 128});
static tensor<float, loc_t::device> V35_w({8192, 8192});
static tensor<float, loc_t::device> V3_data({384, 128});
static tensor<float, loc_t::device> V11_data({384, 128});
static tensor<float, loc_t::device> Attn_mask({1, 1, 384, 384});
static tensor<float, loc_t::device> V38_w({8192, 22016});
static tensor<float, loc_t::device> V40_w({8192, 22016});
static tensor<float, loc_t::device> V42_w({22016, 8192});
static tensor<int64_t, loc_t::device> Position_ids({1, 384});
static tensor<float, loc_t::device> Output({1, 384, 8192});

DEFINE_BFUNC(0)
DEFINE_BFUNC(1)
DEFINE_BFUNC(2)
DEFINE_BFUNC(3)
DEFINE_BFUNC(4)
DEFINE_BFUNC(5)
DEFINE_BFUNC(6)
DEFINE_BFUNC(7)

#define LOAD_FILE(name, i, type)                                               \
    {                                                                          \
        auto src_##name = read_file(std::string(argv[(i)]));                   \
        span_copy(name.data(), gsl::make_span(src_##name).as_span<type>());    \
    }

/**
 * @brief demo2 X.bin WQ.bin WK.bin WV.bin WM.bin
 *
 * @param argc
 * @param argv
 * @return int
 */
int main([[maybe_unused]] int argc, char **argv) {
    // spdlog::set_level(spdlog::level::debug);
    global_hardware_init();

    LOAD_FILE(Hidden_in, 1, float);
    LOAD_FILE(V0_gamma, 2, float);
    LOAD_FILE(V0_beta, 3, float);
    LOAD_FILE(V2_w, 4, float);
    LOAD_FILE(V16_w, 5, float);
    LOAD_FILE(V31_w, 6, float);
    LOAD_FILE(V35_w, 7, float);
    LOAD_FILE(V3_data, 8, float);
    LOAD_FILE(V11_data, 9, float);
    LOAD_FILE(Attn_mask, 10, float);
    LOAD_FILE(V38_w, 11, float);
    LOAD_FILE(V40_w, 12, float);
    LOAD_FILE(V42_w, 13, float);
    LOAD_FILE(Position_ids, 14, int64_t);
    LOAD_FILE(Output, 15, float);

    pthread_t t_0_0, t_1_0, t_2_0, t_3_0, t_4_0, t_5_0, t_6_0, t_7_0;
    pthread_t t_0_1, t_1_1, t_2_1, t_3_1, t_4_1, t_5_1, t_6_1, t_7_1;
    pthread_t t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2, t_6_2, t_7_2;
    pthread_t t_0_3, t_1_3, t_2_3, t_3_3, t_4_3, t_5_3, t_6_3, t_7_3;

    pthread_create(&t_0_0, NULL, f_0_0, NULL);
    pthread_create(&t_0_1, NULL, f_0_1, NULL);
    pthread_create(&t_0_2, NULL, f_0_2, NULL);
    pthread_create(&t_0_3, NULL, f_0_3, NULL);
    pthread_create(&t_1_0, NULL, f_1_0, NULL);
    pthread_create(&t_1_1, NULL, f_1_1, NULL);
    pthread_create(&t_1_2, NULL, f_1_2, NULL);
    pthread_create(&t_1_3, NULL, f_1_3, NULL);
    pthread_create(&t_2_0, NULL, f_2_0, NULL);
    pthread_create(&t_2_1, NULL, f_2_1, NULL);
    pthread_create(&t_2_2, NULL, f_2_2, NULL);
    pthread_create(&t_2_3, NULL, f_2_3, NULL);
    pthread_create(&t_3_0, NULL, f_3_0, NULL);
    pthread_create(&t_3_1, NULL, f_3_1, NULL);
    pthread_create(&t_3_2, NULL, f_3_2, NULL);
    pthread_create(&t_3_3, NULL, f_3_3, NULL);
    pthread_create(&t_4_0, NULL, f_4_0, NULL);
    pthread_create(&t_4_1, NULL, f_4_1, NULL);
    pthread_create(&t_4_2, NULL, f_4_2, NULL);
    pthread_create(&t_4_3, NULL, f_4_3, NULL);
    pthread_create(&t_5_0, NULL, f_5_0, NULL);
    pthread_create(&t_5_1, NULL, f_5_1, NULL);
    pthread_create(&t_5_2, NULL, f_5_2, NULL);
    pthread_create(&t_5_3, NULL, f_5_3, NULL);
    pthread_create(&t_6_0, NULL, f_6_0, NULL);
    pthread_create(&t_6_1, NULL, f_6_1, NULL);
    pthread_create(&t_6_2, NULL, f_6_2, NULL);
    pthread_create(&t_6_3, NULL, f_6_3, NULL);
    pthread_create(&t_7_0, NULL, f_7_0, NULL);
    pthread_create(&t_7_1, NULL, f_7_1, NULL);
    pthread_create(&t_7_2, NULL, f_7_2, NULL);
    pthread_create(&t_7_3, NULL, f_7_3, NULL);

    pthread_join(t_0_0, NULL);
    pthread_join(t_0_1, NULL);
    pthread_join(t_0_2, NULL);
    pthread_join(t_0_3, NULL);
    pthread_join(t_1_0, NULL);
    pthread_join(t_1_1, NULL);
    pthread_join(t_1_2, NULL);
    pthread_join(t_1_3, NULL);
    pthread_join(t_2_0, NULL);
    pthread_join(t_2_1, NULL);
    pthread_join(t_2_2, NULL);
    pthread_join(t_2_3, NULL);
    pthread_join(t_3_0, NULL);
    pthread_join(t_3_1, NULL);
    pthread_join(t_3_2, NULL);
    pthread_join(t_3_3, NULL);
    pthread_join(t_4_0, NULL);
    pthread_join(t_4_1, NULL);
    pthread_join(t_4_2, NULL);
    pthread_join(t_4_3, NULL);
    pthread_join(t_5_0, NULL);
    pthread_join(t_5_1, NULL);
    pthread_join(t_5_2, NULL);
    pthread_join(t_5_3, NULL);
    pthread_join(t_6_0, NULL);
    pthread_join(t_6_1, NULL);
    pthread_join(t_6_2, NULL);
    pthread_join(t_6_3, NULL);
    pthread_join(t_7_0, NULL);
    pthread_join(t_7_1, NULL);
    pthread_join(t_7_2, NULL);
    pthread_join(t_7_3, NULL);

    // auto cos = cosine(QKH.data().begin(),
    //                   gsl::make_span(src_QKH).as_span<float>().begin(),
    //                   QKH.data().size());
    // printf("QKH cosine %f\n", cos);

    return 0;
}