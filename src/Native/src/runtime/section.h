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
#include <nncase/runtime/model.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>

BEGIN_NS_NNCASE_RUNTIME

gsl::span<const gsl::byte>
find_section(const char *name, gsl::span<const gsl::byte> sections) noexcept;
gsl::span<const gsl::byte> read_sections(span_reader &sr,
                                         size_t sections) noexcept;

END_NS_NNCASE_RUNTIME
