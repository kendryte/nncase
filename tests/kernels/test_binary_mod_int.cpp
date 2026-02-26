/* Copyright 2019-2023 Canaan Inc.
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
#include "kernel_test.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;
using namespace nncase::runtime::stackvm;

#if 0
NNCASE_TEST_CLASS(BinaryTest)

NNCASE_TESTSUITE_INIT(BinaryTest, Binary, 2, dt_int32, dt_int64,
                      dims_t{1, 3, 16, 16}, dims_t{3, 16, 16}, dims_t{16, 16},
                      dims_t{16}, dims_t{1})

NNCASE_TEST_BODY(BinaryTest, min, kernels::stackvm::binary, binary_op_t::min, 1,
                 VEC, ortki_Min, orts, sizeof(orts) / sizeof(orts[0]))
NNCASE_TEST_BODY(BinaryTest, max, kernels::stackvm::binary, binary_op_t::max, 1,
                 VEC, ortki_Max, orts, sizeof(orts) / sizeof(orts[0]))
NNCASE_TEST_BODY(BinaryTest, mod, kernels::stackvm::binary, binary_op_t::mod, 1,
                 NORMAL, ortki_Mod, l_ort, r_ort, (long)0)

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif