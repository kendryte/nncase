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

//fixed fixed fixed group, for demonstrate the basic test macro
GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_normal,  
                            (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
                           float, add, Add) 

GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_lhs_scalar,  
                            (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
                           float, add, Add) 

GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_rhs_scalar,  
                            (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1>), (fixed_shape_v<1, 3, 16, 16>),
                           float, add, Add) 

GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_lhs_vector,  
                            (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<1, 3, 16, 16>),
                           float, add, Add) 

GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_rhs_vector,  
                            (fixed_shape_v<1, 3, 16, 16>), (fixed_shape_v<16>), (fixed_shape_v<1, 3, 16, 16>),
                           float, add, Add) 

GENERATE_BINARY_TEST(BinaryTestAddFloat, fixed_fixed_fixed_broadcast_multidirectional,  
                            (fixed_shape_v<1, 3, 1, 16>), (fixed_shape_v<3, 1, 16, 1>), (fixed_shape_v<3, 3, 16, 16>),
                           float, add, Add) 

//fixed dynamic dynamic group(with default shape)
GENERATE_BINARY_TEST_GROUP(BinaryTestAddFloat, fixed, dynamic,dynamic,  
                           float, add, Add) 
//dynamic fixed dynamic group
GENERATE_BINARY_TEST_GROUP(BinaryTestAddFloat, dynamic, fixed, dynamic,  
                           float, add, Add) 
//dynamic dynamic dynamic group
GENERATE_BINARY_TEST_GROUP(BinaryTestAddFloat, dynamic ,dynamic,dynamic,  
                           float, add, Add) 
                           


DEFINE_test_vector(add, Add)

TEST(BinaryTestAddFloat, vector) {                                        
    TEST_VECTOR(float)                                                    
    TEST_VECTOR(int32_t)                                                  
    TEST_VECTOR(int64_t)                                                  
}                                                                          

int main(int argc, char *argv[]) {                                         
    ::testing::InitGoogleTest(&argc, argv);                                
    return RUN_ALL_TESTS();                                                
}

