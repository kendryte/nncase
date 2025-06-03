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

// DEFINE_NTT_BINARY_TEST(add, Add)
    TEST(BinaryTestAddFloat, fixed_fixed_fixed) {                     
        /* init */                                                             
        auto shape = fixed_shape_v<2, 2>;                              
        auto ntt_lhs = make_unique_tensor<float>(shape);                       
        auto ntt_rhs = make_unique_tensor<float>(shape);                       
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           
        printf("ntt_lhs:\n");
        PRINT_TENSOR((*ntt_lhs));
        printf("ntt_rhs:\n");
        PRINT_TENSOR((*ntt_rhs));
        /* ntt */                                                              
        auto ntt_output1 = make_unique_tensor<float>(shape);                   
        ntt::binary<ntt::ops::add>(*ntt_lhs, *ntt_rhs, *ntt_output1);     
        printf("ntt_output1:\n");
        PRINT_TENSOR((*ntt_output1));
                                                                               
        /* ort */                                                              
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             
        auto ort_output = ortki_Add(ort_lhs, ort_rhs);                  
                                                                               
        /* compare */                                                          
        auto ntt_output2 = make_unique_tensor<float>(shape);                   
        NttTest::ort2ntt(ort_output, *ntt_output2);                            
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      
    }                 

// DEBUG_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed,  
//                             (fixed_shape_v<2, 2>), (fixed_shape_v<2, 2>), 
//                            float, add, Add) 


    int main(int argc, char *argv[]) {                                         
        ::testing::InitGoogleTest(&argc, argv);                                
        return RUN_ALL_TESTS();                                                
    }

