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
#include "kernels/cast.h"
#include "kernels/clamp.h"
#include "kernels/compare.h"
#include "kernels/concat.h"
#include "kernels/conv2d.h"
#include "kernels/copy.h"
#include "kernels/expand.h"
#include "kernels/gather.h"
#include "kernels/im2col.h"
#include "kernels/instance_norm.h"
#include "kernels/matmul.h"
#include "kernels/pack.h"
#include "kernels/packed_layer_norm.h"
#include "kernels/packed_softmax.h"
#include "kernels/pad.h"
#include "kernels/reduce.h"
#include "kernels/reduce_arg.h"
#include "kernels/resize_image.h"
#include "kernels/scatter_nd.h"
#include "kernels/slice.h"
#include "kernels/transpose.h"
#include "kernels/unary.h"
#include "kernels/unpack.h"
#include "kernels/where.h"
#include "primitive_ops.h"
#include "tensor.h"
#include "tensor_ops.h"
#include "ukernels.h"
#include "utility.h"
#include "vector.h"

#ifdef __AVX2__
#include "arch/x86_64/arch_types.h"
#include "arch/x86_64/primitive_ops.h"
#include "arch/x86_64/tensor_ops.h"
#include "arch/x86_64/ukernels.h"
#elif __aarch64__
#include "arch/aarch64/arch_types.h"
#include "arch/aarch64/primitive_ops.h"
#include "arch/aarch64/tensor_ops.h"
#elif __riscv_vector
#include "arch/riscv64/arch_types.h"
#include "arch/riscv64/primitive_ops.h"
#include "arch/riscv64/tensor_ops.h"
#endif
