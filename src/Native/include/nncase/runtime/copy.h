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
#include "host_buffer.h"
#include <nncase/runtime/small_vector.hpp>

BEGIN_NS_NNCASE_RUNTIME

result<void> copy(host_buffer_t src_buffer, host_buffer_t dest_buffer,
                  size_t src_start, size_t dest_start, datatype_t datatype,
                  std::span<const size_t> shape,
                  std::span<const size_t> src_strides,
                  std::span<const size_t> dest_strides) noexcept;

END_NS_NNCASE_RUNTIME
