#include "cluster_def.h"
#include <chrono>
#include <io_utils.h>
#include <runtime_utils.h>
// #include <pthread.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {                                             \
        block##b::thread##t::stage1_kernel(                                    \
            Hidden_in, V0_gamma, V0_beta, V2_w, V16_w, V31_w, V35_w, V3_data,  \
            V11_data, Attn_mask, V38_w, V40_w, V42_w, Position_ids, Output,    \
            V25, GV31, ImmOutputs);                                            \
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
static tensor<float, loc_t::device> V25({1, 64, 128, 384});
static tensor<float, loc_t::device> GV31({1, 64, 384, 128});

constexpr int OutNum = 11;
static tensor<float, loc_t::device> goldenImmOutputs[OutNum] = {
    tensor<float, loc_t::device>({1, 384, 8192}),    // v0
    tensor<float, loc_t::device>({1, 64, 384, 128}), // v2
    tensor<float, loc_t::device>({1, 64, 384, 128}), // v10
    tensor<float, loc_t::device>({1, 64, 384, 128}), // v14
    tensor<float, loc_t::device>({1, 384, 128}),     // v3
    tensor<float, loc_t::device>({1, 64, 384, 128}), // v16
    tensor<float, loc_t::device>({1, 64, 384, 384}), // v28
    tensor<float, loc_t::device>({1, 64, 384, 128}), // v32
    tensor<float, loc_t::device>({1, 384, 8192}),    // v35
    tensor<float, loc_t::device>({1, 384, 22016}),   // v38
    tensor<float, loc_t::device>({1, 384, 8192}),    // v43
};

static tensor<float, loc_t::device> ImmOutputs[OutNum] = {
    tensor<float, loc_t::device>({1, 384, 8192}),
    tensor<float, loc_t::device>({1, 64, 384, 128}),
    tensor<float, loc_t::device>({1, 64, 384, 128}),
    tensor<float, loc_t::device>({1, 64, 384, 128}),
    tensor<float, loc_t::device>({1, 384, 128}),
    tensor<float, loc_t::device>({1, 64, 384, 128}),
    tensor<float, loc_t::device>({1, 64, 384, 384}),
    tensor<float, loc_t::device>({1, 64, 384, 128}),
    tensor<float, loc_t::device>({1, 384, 8192}),
    tensor<float, loc_t::device>({1, 384, 22016}),
    tensor<float, loc_t::device>({1, 384, 8192}),
};

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

#define CLEAR_SHARED(i)                                                        \
    tdma_fill_async<float>(block##i::shared::V2, 0);                           \
    tdma_fill_async<float>(block##i::shared::V5, 0);                           \
    tdma_fill_async<float>(block##i::shared::V13, 0);                          \
    tdma_fill_async<float>(block##i::shared::V16, 0);                          \
    tdma_fill_async<float>(block##i::shared::V25, 0);                          \
    tdma_fill_async<float>(block##i::shared::V26, 0);                          \
    tdma_fill_async<float>(block##i::shared::V31, 0);                          \
    tdma_fill_async<float>(block##i::shared::V32, 0);                          \
    tdma_fill_async<float>(block##i::shared::V33, 0);                          \
    tdma_fill_async<float>(block##i::shared::V35, 0);                          \
    tdma_fill_async<float>(block##i::shared::V38, 0);                          \
    tdma_fill_async<float>(block##i::shared::V40, 0);                          \
    tdma_fill_async<float>(block##i::shared::V42, 0);

/**
 * @brief demo2 X.bin WQ.bin WK.bin WV.bin WM.bin
 *
 * @param argc
 * @param argv
 * @return int
 */
int main([[maybe_unused]] int argc, char **argv) {
    global_hardware_init();
    LOAD_FILE(Hidden_in, 1, float);
    LOAD_FILE(Attn_mask, 10, float);
    LOAD_FILE(Position_ids, 14, int64_t);
    for (auto index = 0; index < 80; index++) {
        // spdlog::set_level(spdlog::level::debug);
        auto start = std::chrono::steady_clock::now();

        LOAD_FILE(V0_gamma, 2, float);
        LOAD_FILE(V0_beta, 3, float);
        LOAD_FILE(V2_w, 4, float);
        LOAD_FILE(V16_w, 5, float);
        LOAD_FILE(V31_w, 6, float);
        LOAD_FILE(V35_w, 7, float);
        LOAD_FILE(V3_data, 8, float);
        LOAD_FILE(V11_data, 9, float);
        LOAD_FILE(V38_w, 11, float);
        LOAD_FILE(V40_w, 12, float);
        LOAD_FILE(V42_w, 13, float);

        for (auto o = 0; o < argc - 16; o++) {
            auto src_output = read_file(std::string(argv[(16 + o)]));
            span_copy(goldenImmOutputs[o].data(),
                      gsl::make_span(src_output).as_span<float>());
        }

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

        auto stop = std::chrono::steady_clock::now();
        double duration =
            std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << "Demo run: " << duration / 1000 << " secs" << std::endl;
        
        for (auto o = 0; o < argc - 16; o++) {
            auto cos = cosine(ImmOutputs[o].data().begin(),
                              goldenImmOutputs[o].data().begin(),
                              goldenImmOutputs[o].data().size());
            printf("%s cosine %f\n", argv[16 + o], cos);
        }
        {
            auto src_output = read_file(std::string(argv[15]));
            auto src_span = gsl::make_span(src_output).as_span<float>();
            auto cos = cosine(Hidden_in.data().data(), src_span.data(),
                              src_span.size());
            printf("Output cosine %f\n", cos);
        }

        CLEAR_SHARED(0);
        CLEAR_SHARED(1);
        CLEAR_SHARED(2);
        CLEAR_SHARED(3);
        CLEAR_SHARED(4);
        CLEAR_SHARED(5);
        CLEAR_SHARED(6);
        CLEAR_SHARED(7);
    }

    return 0;
}