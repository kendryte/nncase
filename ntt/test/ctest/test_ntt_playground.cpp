/* Copyright 2019-2024 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "test_ntt_binary.h"
#include <nncase/half.h>

//test case combination:
// 1. lhs/rhs
// 2. dynamic/fixed
// 3. lhs broadcast to rhs, rhs broadcast to lhs
// 3.1. 1 dim broadcast
// 3.2. 2 dims broadcast
// 4. scalar/vector/2d vector
// 5. tensor/ view

// TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_vector) {
//     // init
//     auto ntt_tensor_lhs =  make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1>);
//     NttTest::init_tensor(ntt_tensor_lhs, -10, 10);

//     auto ntt_tensor_rhs =  make_tensor<int>(ntt::fixed_shape_v<1, 3, 1, 16>);
//     NttTest::init_tensor(ntt_tensor_rhs, -10, 10);

//     // ntt
//     auto ntt_output1 = make_tensor<ntt::vector<int, 8>>(ntt::fixed_shape_v<1, 3, 1, 16>);
//     ntt::binary<ntt::ops::add>(ntt_tensor_lhs, ntt_tensor_rhs, ntt_output1);

//     // // if mxn tensor-of-vector<v> op mxn tensor-of-scalar, 
//     // //broadcast the ntt mxn tensor-of-scalar to ort mxnxv tensor-of-scalar

//     // // ort
//     auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_tensor_lhs, ntt_tensor_rhs);
//     auto ort_output = ortki_Add(ort_input_lhs, ort_input_rhs);
//     // ortki_Add(ort_input_lhs, ort_input_rhs);
//     // // compare
//     auto ntt_output2 = make_tensor<ntt::vector<uint32_t, 8>>(ntt::fixed_shape_v<1, 3, 1, 16>);
//     NttTest::ort2ntt(ort_output, ntt_output2);
//     EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

// }



TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_1D_vector_rhs_2D_vector) {
    // init
    auto ntt_tensor_lhs =  make_tensor<ntt::vector<double, 8>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_tensor_lhs, 0, 100000);

    auto ntt_tensor_rhs =  make_tensor<ntt::vector<double, 8>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_tensor_rhs, 0, 100000000);

    // ntt
    auto ntt_output1 = make_tensor<ntt::vector<double, 8>>(ntt::fixed_shape_v<1>);
    ntt::binary<ntt::ops::pow>(ntt_tensor_lhs, ntt_tensor_rhs, ntt_output1);

    // // if mxn tensor-of-vector<v> op mxn tensor-of-scalar, 
    // //broadcast the ntt mxn tensor-of-scalar to ort mxnxv tensor-of-scalar

    // // ort

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_tensor_lhs, ntt_tensor_rhs);
    // auto ntt_max = make_tensor<float>(ntt::fixed_shape_v<1>);
    // ntt_max(0) = 2.40614e+38;
    // auto ort_max = NttTest::ntt2ort(ntt_max);


    // auto ntt_zero = make_tensor<float>(ntt::fixed_shape_v<1>);
    // ntt_zero(0) = 0.0f;
    // auto ort_zero = NttTest::ntt2ort(ntt_zero);

    // auto ort_output = ortki_Div(ortki_Add(ortki_Add(ort_input_rhs,ort_neg1), ort_input_lhs), ort_input_rhs);
    // const size_t num_inputs = 2;
    // ortki::OrtKITensor* input_tensors[num_inputs];
    // input_tensors[0] = ort_input_lhs;
    // input_tensors[1] = ort_input_rhs;
    // auto ort_output = ortki_Min(input_tensors, num_inputs);
    // auto ort_output = ortki_Clip(ortki_Pow(ort_input_lhs, ort_input_rhs), ort_zero, ort_max);
    auto ort_output = ortki_Pow(ort_input_lhs, ort_input_rhs);

    // // compare
    auto ntt_output2 = make_tensor<ntt::vector<double, 8>>(ntt::fixed_shape_v<1>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

}

TEST(BinaryTestAddint, are_close_fp16_behavior) {
    using namespace nncase;
    
    // Test specific fp16 cases that are returning false unexpectedly
    std::cout << "Testing are_close behavior for fp16 (half) values:" << std::endl;
    
    // Test case 1: lhs = 922, rhs = 922.5
    {
        half lhs(5.1875f);
        half rhs(5.16406);
        
        // Test are_close directly on half types
        bool result = NttTest::are_close(lhs, rhs);
        
        // Also test float conversion for comparison
        float lhs_f = static_cast<float>(lhs);
        float rhs_f = static_cast<float>(rhs);
        std::cout << "Test 1 - lhs: " << lhs_f << " (fp16:5.1875), rhs: " << rhs_f << " (fp16:5.16406)" 
                  << ", are_close result: " << (result ? "true" : "false")
                  << ", diff: " << std::abs(lhs_f - rhs_f) << std::endl;
    }
    
    // Test case 2: lhs = -59.1875, rhs = -59.2188
    {
        half lhs(1.4375f);
        half rhs(1.43652f);
        
        bool result = NttTest::are_close(lhs, rhs);
        
        float lhs_f = static_cast<float>(lhs);
        float rhs_f = static_cast<float>(rhs);
        std::cout << "Test 2 - lhs: " << lhs_f << " (fp16: 1.4375f), rhs: " << rhs_f << " (fp16: 1.43652)" 
                  << ", are_close result: " << (result ? "true" : "false")
                  << ", diff: " << std::abs(lhs_f - rhs_f) << std::endl;
    }
    
    // Test case 3: lhs = -7192, rhs = -7196
    {
        half lhs(321.5f);
        half rhs(321.25f);
        
        bool result = NttTest::are_close(lhs, rhs);
        
        float lhs_f = static_cast<float>(lhs);
        float rhs_f = static_cast<float>(rhs);
        std::cout << "Test 3 - lhs: " << lhs_f << " (fp16: 321.5), rhs: " << rhs_f << " (fp16: 321.25)" 
                  << ", are_close result: " << (result ? "true" : "false")
                  << ", diff: " << std::abs(lhs_f - rhs_f) << std::endl;
    }
    
    // Test case 4: lhs = 6996, rhs = 6992
    {
        half lhs(6996.0f);
        half rhs(6992.0f);
        
        bool result = NttTest::are_close(lhs, rhs);
        
        float lhs_f = static_cast<float>(lhs);
        float rhs_f = static_cast<float>(rhs);
        std::cout << "Test 4 - lhs: " << lhs_f << " (fp16: 6996), rhs: " << rhs_f << " (fp16: 6992)" 
                  << ", are_close result: " << (result ? "true" : "false")
                  << ", diff: " << std::abs(lhs_f - rhs_f) << std::endl;
    }
    
    // Test with different tolerance values for fp16
    {
        half lhs(922.0f);
        half rhs(922.5f);
        
        bool result_default = NttTest::are_close(lhs, rhs);
        bool result_loose = NttTest::are_close(lhs, rhs, 1.0, 1e-3);  // More loose tolerance for fp16
        bool result_tight = NttTest::are_close(lhs, rhs, 1e-12, 1e-9); // Tighter tolerance
        
        float lhs_f = static_cast<float>(lhs);
        float rhs_f = static_cast<float>(rhs);
        
        std::cout << "\nTolerance test for fp16 - lhs: " << lhs_f << ", rhs: " << rhs_f << std::endl;
        std::cout << "  Default tolerance (1e-9, 1e-5): " << (result_default ? "true" : "false") << std::endl;
        std::cout << "  Loose tolerance (1.0, 1e-3): " << (result_loose ? "true" : "false") << std::endl;
        std::cout << "  Tight tolerance (1e-12, 1e-9): " << (result_tight ? "true" : "false") << std::endl;
        
        // Calculate what the actual tolerance check values are
        double abs_diff = std::abs(lhs_f - rhs_f);
        double rel_tol_default = 1e-5 * std::max(std::abs(lhs_f), std::abs(rhs_f));
        double abs_tol_default = 1e-9;
        double threshold_default = std::max(abs_tol_default, rel_tol_default);
        
        std::cout << "  Absolute difference: " << abs_diff << std::endl;
        std::cout << "  Default relative tolerance: " << rel_tol_default << std::endl;
        std::cout << "  Default absolute tolerance: " << abs_tol_default << std::endl;
        std::cout << "  Default threshold: " << threshold_default << std::endl;
        std::cout << "  Passes default threshold: " << (abs_diff <= threshold_default ? "true" : "false") << std::endl;
    }
    
    // Test fp16 precision limitations
    {
        std::cout << "\n=== FP16 Precision Analysis ===" << std::endl;
        
        // Show actual fp16 values after conversion
        half h1(922.0f);
        half h2(922.5f);
        float f1 = static_cast<float>(h1);
        float f2 = static_cast<float>(h2);
        
        std::cout << "Input: 922.0 -> fp16 -> float: " << f1 << std::endl;
        std::cout << "Input: 922.5 -> fp16 -> float: " << f2 << std::endl;
        std::cout << "Difference after fp16 conversion: " << std::abs(f1 - f2) << std::endl;
        
        // Show raw fp16 values
        std::cout << "Raw fp16 value for 922.0: 0x" << std::hex << h1.raw() << std::dec << std::endl;
        std::cout << "Raw fp16 value for 922.5: 0x" << std::hex << h2.raw() << std::dec << std::endl;
    }
}


// //fixed fixed fixed group, for demonstrate the basic test macro
// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_normal,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_scalar,  
//                             (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_rhs_scalar,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_lhs_vector,  
//                             (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_rhs_vector,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>),
//                            int, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddint, fixed_fixed_fixed_broadcast_multidirectional,  
//                             (fixed_shape_v<1, 3, 1, 16>), (fixed_shape_v<3, 1, 16, 1>), (fixed_shape_v<3, 3, 16, 16>),
//                            int, add, Add) 

// //fixed dynamic dynamic group(with default shape)
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddint, fixed, dynamic,dynamic,  
//                            int, add, Add) 
// //dynamic fixed dynamic group
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddint, dynamic, fixed, dynamic,  
//                            int, add, Add) 
// //dynamic dynamic dynamic group
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddint, dynamic ,dynamic,dynamic,  
//                            int, add, Add) 
                           


// DEFINE_test_vector(add, Add)
// TEST(BinaryTestAddint, vector) {                                        
//     TEST_VECTOR(int)                                                    
//     TEST_VECTOR(int32_t)                                                  
//     TEST_VECTOR(int64_t)                                                  
// }                                                                          

int main(int argc, char *argv[]) {                                         
    ::testing::InitGoogleTest(&argc, argv);                                
    return RUN_ALL_TESTS();                                                
}

