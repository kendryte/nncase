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
#include "compiler_defs.h"
#include <nncase/runtime/error.h>

BEGIN_NS_NNCASE_RT_K210

enum class nncase_k210_errc
{
    k210_illegal_instruction = 0x01
};

NNCASE_MODULES_K210_API const std::error_category &nncase_k210_category() noexcept;
NNCASE_MODULES_K210_API std::error_condition make_error_condition(nncase_k210_errc code);

END_NS_NNCASE_RT_K210

namespace std
{
template <>
struct is_error_condition_enum<nncase::runtime::k210::nncase_k210_errc> : true_type
{
};
}
