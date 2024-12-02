#include "ntt_test.h"
#include <iomanip>
#include <memory>
#include <nncase/ntt/ntt.h>

using namespace nncase;

namespace nncase::ntt {
    template <>
    struct vector_storage_traits<float> {
        using buffer_type = float;
    };
}

template <typename T, size_t M, size_t P>
void benchmark_ntt_expand_nopack(T init_low, T init_high) {
    std::string pack_mode = "NoPack";
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 20000;
#else
    constexpr size_t run_size = 20000;
#endif

    using in_tensor_type = ntt::tensor<T, ntt::fixed_shape<M>>;
    using out_tensor_type = ntt::tensor<T, ntt::fixed_shape<M, P>>;

    std::unique_ptr<in_tensor_type> ntt_input(new in_tensor_type);
    std::unique_ptr<out_tensor_type> ntt_output(new out_tensor_type);

    NttTest::init_tensor(*ntt_input, init_low, init_high);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::expand(*ntt_input, *ntt_output);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::expand(*ntt_input, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();
    std::cout << __FUNCTION__ << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / M / run_size << " cycles"
              << std::endl;
}

template <typename T, size_t M, size_t N, size_t P>
void benchmark_ntt_expand_nopack1(T init_low, T init_high) {
    std::string pack_mode = "NoPack";
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 20000;
#else
    constexpr size_t run_size = 20000;
#endif

    using in_tensor_type = ntt::tensor<T, ntt::fixed_shape<M, N>>;
    using out_tensor_type = ntt::tensor<T, ntt::fixed_shape<M, P>>;

    std::unique_ptr<in_tensor_type> ntt_input(new in_tensor_type);
    std::unique_ptr<out_tensor_type> ntt_output(new out_tensor_type);

    NttTest::init_tensor(*ntt_input, init_low, init_high);

    // warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::expand(*ntt_input, *ntt_output);

    // run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::expand(*ntt_input, *ntt_output);
        asm volatile("" ::"g"(ntt_output));
    }
    auto t2 = NttTest::get_cpu_cycle();
    std::cout << __FUNCTION__ << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / M / run_size << " cycles"
              << std::endl;
}

template <typename T, size_t M, size_t N, size_t P, size_t VLEN>
void benchmark_ntt_expand_pack(T init_low, T init_high) {
    std::string pack_mode = "Pack";
    constexpr size_t warmup_size = 10;
#if __riscv
    constexpr size_t run_size = 300;
#elif __x86_64__
    constexpr size_t run_size = 20000;
#else
    constexpr size_t run_size = 20000;
#endif

    using in_tensor_type = ntt::tensor<T, ntt::fixed_shape<M, N>>;
    using out_tensor_type = ntt::tensor<T, ntt::fixed_shape<M, P>>;
    using packed_in_tensor_type = ntt::tensor<ntt::vector<T, VLEN>, ntt::fixed_shape<M / VLEN, N>>;
    using packed_out_tensor_type = ntt::tensor<ntt::vector<T, VLEN>, ntt::fixed_shape<M / VLEN, P>>;

    std::unique_ptr<in_tensor_type> ntt_input(new in_tensor_type);
    std::unique_ptr<out_tensor_type> ntt_output(new out_tensor_type);
    std::unique_ptr<packed_in_tensor_type> packed_input(new packed_in_tensor_type);
    std::unique_ptr<packed_out_tensor_type> packed_output(new packed_out_tensor_type);
    
    NttTest::init_tensor(*ntt_input, init_low, init_high);

    // Pack the input tensor
    ntt::pack<0>(*ntt_input, *packed_input);

    // Warm up
    for (size_t i = 0; i < warmup_size; i++)
        ntt::expand(*packed_input, *packed_output);

    // Run
    auto t1 = NttTest::get_cpu_cycle();
    for (size_t i = 0; i < run_size; i++) {
        ntt::expand(*packed_input, *packed_output);
        asm volatile("" ::"g"(packed_output));
    }
    auto t2 = NttTest::get_cpu_cycle();

    std::cout << __FUNCTION__ << "_" << pack_mode << " took "
              << std::setprecision(1) << std::fixed
              << static_cast<float>(t2 - t1) / (M / VLEN) / run_size << " cycles"
              << std::endl;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    constexpr size_t M1 = 1;
    constexpr size_t P1 = 2;
    benchmark_ntt_expand_nopack<float, M1, P1>(-10.f, 10.f);

    constexpr size_t M2 = 1024;
    constexpr size_t N2 = 1;
    constexpr size_t P2 = 2048;
    benchmark_ntt_expand_nopack1<float, M2, N2, P2>(-10.f, 10.f);

    constexpr size_t M3 = 32;
    constexpr size_t N3 = 1;
    constexpr size_t P3 = 2;
    constexpr size_t VLEN3 = 4;
    benchmark_ntt_expand_pack<float, M3, N3, P3, VLEN3>(-10.f, 10.f);

    return 0;
}