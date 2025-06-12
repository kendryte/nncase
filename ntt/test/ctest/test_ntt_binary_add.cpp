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

//test case combination:
// 1. lhs/rhs
// 2. dynamic/fixed
// 3. lhs broadcast to rhs, rhs broadcast to lhs
// 4. scalar/vector


TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_lhs_vector) {
    // init
    auto ntt_tensor_lhs =  make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_tensor_lhs, -10.f, 10.f);

    auto ntt_tensor_rhs =  make_tensor<float>(ntt::fixed_shape_v<1, 3, 16, 16>);
    NttTest::init_tensor(ntt_tensor_rhs, -10.f, 10.f);

    // ntt
    auto ntt_output1 = make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<1, 3, 16, 16>);
    ntt::binary<ntt::ops::add>(ntt_tensor_lhs, ntt_tensor_rhs, ntt_output1);

    // // ort
    // auto ort_lhs = NttTest::ntt2ort(ntt_tensor_lhs);
    // auto ort_rhs = NttTest::ntt2ort(ntt_tensor_rhs);
    // auto ort_output = ortki_Add(ort_lhs, ort_rhs);

    // // compare
    // auto ntt_output2 = make_unique_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<1, 3, 16, 16>);
    // NttTest::ort2ntt(ort_output, ntt_output2);
    // EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

}

// //fixed fixed fixed group, for demonstrate the basic test macro
// GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_normal,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            float, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_lhs_scalar,  
//                             (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            float, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_rhs_scalar,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>),
//                            float, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_lhs_vector,  
//                             (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
//                            float, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_rhs_vector,  
//                             (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>),
//                            float, add, Add) 

// GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_multidirectional,  
//                             (fixed_shape_v<1, 3, 1, 16>), (fixed_shape_v<3, 1, 16, 1>), (fixed_shape_v<3, 3, 16, 16>),
//                            float, add, Add) 

// //fixed dynamic dynamic group(with default shape)
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddFloat, fixed, dynamic,dynamic,  
//                            float, add, Add) 
// //dynamic fixed dynamic group
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddFloat, dynamic, fixed, dynamic,  
//                            float, add, Add) 
// //dynamic dynamic dynamic group
// GENERATE_BINARY_TEST_GROUP(BinaryTestAddFloat, dynamic ,dynamic,dynamic,  
//                            float, add, Add) 
                           


// DEFINE_test_vector(add, Add)
// TEST(BinaryTestAddFloat, vector) {                                        
//     TEST_VECTOR(float)                                                    
//     TEST_VECTOR(int32_t)                                                  
//     TEST_VECTOR(int64_t)                                                  
// }                                                                          

int main(int argc, char *argv[]) {                                         
    ::testing::InitGoogleTest(&argc, argv);                                
    return RUN_ALL_TESTS();                                                
}

