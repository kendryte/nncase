#include "cluster_def.h"
#include <io_utils.h>
#include <runtime_utils.h>
// #include <pthread.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {                                             \
        block##b::thread##t::stage1_kernel(WK, K, Sum);                        \
        return arg;                                                            \
    }

#define DEFINE_BFUNC(b)                                                        \
    DEFINE_TFUNC(b, 0)                                                         \
    DEFINE_TFUNC(b, 1)                                                         \
    DEFINE_TFUNC(b, 2)                                                         \
    DEFINE_TFUNC(b, 3)

tensor<float, loc_t::device> WK({64, 8192, 128});
tensor<float, loc_t::device> K({64, 384, 128});
tensor<float, loc_t::device> Sum({128});

DEFINE_BFUNC(0)
DEFINE_BFUNC(1)
DEFINE_BFUNC(2)
DEFINE_BFUNC(3)
DEFINE_BFUNC(4)
DEFINE_BFUNC(5)
DEFINE_BFUNC(6)
DEFINE_BFUNC(7)

/**
 * @brief demo1 X.bin WK.bin K.bin Sum.bin
 *
 * @param argc
 * @param argv
 * @return int
 */
int main([[maybe_unused]] int argc, char **argv) {
    assert(argc == 5);
    spdlog::set_level(spdlog::level::debug);
    global_hardware_init();

    /* fill tensor */
    auto src_X = read_file(std::string(argv[1]));
    auto src_WK = read_file(std::string(argv[2]));
    auto src_K = read_file(std::string(argv[3]));
    auto src_Sum = read_file(std::string(argv[4]));
    span_copy(WK.data(), gsl::make_span(src_WK).as_span<float>());
    span_copy(block0::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block1::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block2::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block3::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block4::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block5::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block6::shared::X.data(), gsl::make_span(src_X).as_span<float>());
    span_copy(block7::shared::X.data(), gsl::make_span(src_X).as_span<float>());

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

    auto cos =
        cosine(K.data().begin(), gsl::make_span(src_K).as_span<float>().begin(),
               K.data().size());
    printf("K cosine %f\n", cos);

    auto cos2 = cosine(Sum.data().begin(),
                       gsl::make_span(src_Sum).as_span<float>().begin(),
                       Sum.data().size());
    printf("Sum cosine %f\n", cos2);
    return 0;
}