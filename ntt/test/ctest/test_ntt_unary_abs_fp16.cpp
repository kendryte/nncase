#include "ntt_test.h"
#include <gtest/gtest.h>
#include <nncase/ntt/ntt.h>
#include  <nncase/half.h>
using namespace nncase;
// TEST(UnaryTestAbsFloat16,fixed_fixed) {
//     using shape  = ntt::fixed_shape<1, 3, 16, 16>;
//     using tensor_type = ntt::tensor<half, shape>;
//     std::unique_ptr<tensor_type> ntt_input(new tensor_type);
//     NttTest::init_tensor(*ntt_input, -10.f, 10.f);

//     std::unique_ptr<tensor_type> ntt_output1(new tensor_type);

//     ntt::unary<ntt::ops::abs>(*ntt_input, *ntt_output1);
//     std::cout<<"ntt_output: "<<*ntt_output1->elements().begin()<<std::endl;
//     std::cout<<"ntt_input: "<<*ntt_input->elements().begin()<<std::endl;
// }

TEST(UnaryTestAbs, v) {
    ntt::vector<half, NTT_VLEN / (sizeof(half) * 8)> ntt_input;
    NttTest::init_tensor(ntt_input, static_cast<half>(-10), static_cast<half>(10));
    auto ntt_output1 = ntt::abs(ntt_input);
    std::cout << ntt_output1(0) << std::endl;
    std::cout << ntt_input(0) << std::endl;
    ;
}

