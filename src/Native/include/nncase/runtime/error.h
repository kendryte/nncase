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
#include "../compiler_defs.h"
#include <system_error>

BEGIN_NS_NNCASE_RUNTIME

enum class nncase_errc {
    invalid_model_indentifier = 0x01,
    invalid_model_checksum = 0x02,
    invalid_model_version = 0x03,
    runtime_not_found = 0x04,
    datatype_mismatch = 0x05,
    shape_mismatch = 0x06,
    invalid_memory_location = 0x07,
    runtime_register_not_found = 0x08,
    stackvm_illegal_instruction = 0x0100,
    stackvm_illegal_target = 0x0101,
    stackvm_stack_overflow = 0x0102,
    stackvm_stack_underflow = 0x0103,
    stackvm_unknow_custom_call = 0x0104,
    stackvm_duplicate_custom_call = 0x0105,
    nnil_illegal_instruction = 0x0200,
};

NNCASE_API const std::error_category &nncase_category() noexcept;
NNCASE_API std::error_code make_error_code(nncase_errc code);
NNCASE_API std::error_condition make_error_condition(nncase_errc code);

END_NS_NNCASE_RUNTIME

namespace std {
template <>
struct is_error_condition_enum<nncase::runtime::nncase_errc> : true_type {};
} // namespace std
