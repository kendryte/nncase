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
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>
#include <nncase/runtime/stream_reader.h>
#include <nncase/type.h>

BEGIN_NS_NNCASE_RUNTIME

typedef enum : uint8_t {
    type_sig_invalid,
    type_sig_any,
    type_sig_tensor,
    type_sig_tuple,
    type_sig_callable,
    type_sig_end = 0xFF
} type_signature_token_t;

result<type> deserialize_type(span_reader &sr) noexcept;
result<datatype_t> deserialize_datatype(span_reader &sr) noexcept;

result<type> deserialize_type(stream_reader &sr) noexcept;
result<datatype_t> deserialize_datatype(stream_reader &sr) noexcept;

END_NS_NNCASE_RUNTIME
