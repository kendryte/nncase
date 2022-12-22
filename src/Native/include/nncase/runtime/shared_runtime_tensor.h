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
#include "runtime_tensor_impl.h"

BEGIN_NS_NNCASE_RUNTIME

namespace detail {
class host_runtime_tensor_impl;
}

END_NS_NNCASE_RUNTIME

#ifndef NNCASE_SHARED_RUNTIME_TENSOR_PLATFORM_HEADER
#include "shared_runtime_tensor.platform.h"
#else
#include NNCASE_SHARED_RUNTIME_TENSOR_PLATFORM_HEADER
#endif
