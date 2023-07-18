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
#include "macro_util.h"
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

NNCASE_TEST_CLASS(BinaryTest)

NNCASE_TESTSUITE_INIT(BinaryTest, Binary, 1, dt_boolean,
                      GET_DEFAULT_TEST_SHAPE())

NNCASE_TEST_BODY(BinaryTest, logical_and, kernels::stackvm::binary,
                 binary_op_t::logical_and, 1, NORMAL, ortki_And, l_ort, r_ort)
NNCASE_TEST_BODY(BinaryTest, logical_or, kernels::stackvm::binary,
                 binary_op_t::logical_or, 1, NORMAL, ortki_Or, l_ort, r_ort)
NNCASE_TEST_BODY(BinaryTest, logical_xor, kernels::stackvm::binary,
                 binary_op_t::logical_xor, 1, NORMAL, ortki_Xor, l_ort, r_ort)

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}