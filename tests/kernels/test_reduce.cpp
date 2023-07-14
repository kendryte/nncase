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

// NNCASE_TEST_CLASS_ARGS_4(ReduceTest)

// // NNCASE_TESTSUITE_INIT(ReduceTest, Reduce, 3, GET_DEFAULT_TEST_TYPE_3(), GET_DEFAULT_TEST_SHAPE())

// NNCASE_TESTSUITE_INIT_COMBINE(ReduceTest, Reduce, 
//     GET_DEFAULT_TEST_TYPE_3(), 
//     GET_DEFAULT_TEST_INTTYPE(), 
//     GET_DEFAULT_TEST_INTTYPE(), 
//     GET_DEFAULT_TEST_INTTYPE(), 
//     GET_DEFAULT_TEST_SHAPE(), 
//     GET_DEFAULT_TEST_VECTOR_SHAPE(), 
//     GET_DEFAULT_TEST_SCALAR_SHAPE(),
//     GET_DEFAULT_TEST_SCALAR_SHAPE(), 
//     )

// NNCASE_TEST_BODY_ARGS_4(ReduceTest, reduce_max, kernels::stackvm::reduce, reduce_op_t::max, 1, NORMAL, ortki_ReduceMax, a_ort, b_ort, c_ort[0], d_ort[0])


