/* Copyright 2019-2021 Canaan Inc.
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
#pragma once
#include "kernels/binary.h"
#include "kernels/copy.h"
#include "kernels/matmul.h"
#include "kernels/pack.h"
#include "kernels/packed_layer_norm.h"
#include "kernels/packed_matmul.h"
#include "kernels/packed_softmax.h"
#include "kernels/unary.h"
#include "kernels/unpack.h"
#include "tensor.h"
#include "utility.h"
#include "vector_type.h"