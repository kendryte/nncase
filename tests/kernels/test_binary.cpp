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

NNCASE_TEST_CLASS_ARGS_2_ATTR_0(BinaryTest, RANDOM, RANDOM)
NNCASE_TEST_CLASS_ARGS_2_ATTR_0(BinaryLogicTest, RANDOM, RANDOM)

NNCASE_TESTSUITE_INIT_ARGS_2(BinaryTest, Binary, GET_DEFAULT_TEST_TYPE_(),
                             GET_DEFAULT_TEST_TYPE_(),
                             GET_DEFAULT_TEST_SHAPE_(),
                             GET_DEFAULT_TEST_SHAPE_())
NNCASE_TESTSUITE_INIT_ARGS_2(BinaryLogicTest, BinaryLogic,
                             GET_DEFAULT_TEST_BOOL_TYPE_(),
                             GET_DEFAULT_TEST_BOOL_TYPE_(),
                             GET_DEFAULT_TEST_SHAPE_(),
                             GET_DEFAULT_TEST_SHAPE_())

NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryTest, add, kernels::stackvm::binary,
                               binary_op_t::add, ortki_Add, _typecode_0)
NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryTest, sub, kernels::stackvm::binary,
                               binary_op_t::sub, ortki_Sub, _typecode_0)
NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryTest, mul, kernels::stackvm::binary,
                               binary_op_t::mul, ortki_Mul, _typecode_0)
NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryTest, pow, kernels::stackvm::binary,
                               binary_op_t::pow, ortki_Pow, _typecode_0)
NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryLogicTest, logical_and,
                               kernels::stackvm::binary,
                               binary_op_t::logical_and, ortki_And, _typecode_0)
NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryLogicTest, logical_or,
                               kernels::stackvm::binary,
                               binary_op_t::logical_or, ortki_Or, _typecode_0)
NNCASE_TEST_BODY_ARGS_2_ATTR_0(BinaryLogicTest, logical_xor,
                               kernels::stackvm::binary,
                               binary_op_t::logical_xor, ortki_Xor, _typecode_0)

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}