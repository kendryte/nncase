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
#include "macro_util.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;
using namespace nncase::runtime::stackvm;


NNCASE_TEST_CLASS(CompareTest)

NNCASE_TESTSUITE_INIT(CompareTest, Compare, 3,  dt_float32, dt_int32, dt_int64, dims_t{1}, dims_t{16}, dims_t{1, 16}, dims_t{1, 16, 16}, dims_t{3, 3, 1, 16})

NNCASE_TEST_BODY(CompareTest, not_equal, kernels::stackvm::compare, compare_op_t::not_equal, 2, NORMAL, ortki_Not, ortki_Equal, l_ort, r_ort)
NNCASE_TEST_BODY(CompareTest, equal, kernels::stackvm::compare, compare_op_t::equal, 1, NORMAL, ortki_Equal, l_ort, r_ort)
NNCASE_TEST_BODY(CompareTest, greater_or_equal, kernels::stackvm::compare, compare_op_t::greater_or_equal, 1, NORMAL, ortki_GreaterOrEqual, l_ort, r_ort)
NNCASE_TEST_BODY(CompareTest, greater_than, kernels::stackvm::compare, compare_op_t::greater_than, 1, NORMAL, ortki_Greater, l_ort, r_ort)
NNCASE_TEST_BODY(CompareTest, lower_or_equal, kernels::stackvm::compare, compare_op_t::lower_or_equal, 1, NORMAL, ortki_LessOrEqual, l_ort, r_ort)
NNCASE_TEST_BODY(CompareTest, lower_than, kernels::stackvm::compare, compare_op_t::lower_than, 1, NORMAL, ortki_Less, l_ort, r_ort)

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}