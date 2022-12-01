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
#include <nncase/compiler_defs.h>

extern "C" {
struct clr_object_t {};
typedef clr_object_t *clr_object_handle;

NNCASE_API int nncase_compiler_initialize(const char *root_assembly_path);

NNCASE_API int
nncase_compiler_compile_options_create(clr_object_handle *handle);
NNCASE_API int nncase_compiler_compile_options_set_input_file(
    clr_object_handle compile_options, const char *input_file,
    size_t input_file_length);
NNCASE_API int nncase_compiler_compile_options_set_input_format(
    clr_object_handle compile_options, const char *input_format,
    size_t input_format_length);
NNCASE_API int
nncase_compiler_compile_options_set_target(clr_object_handle compile_options,
                                           const char *target,
                                           size_t target_length);
NNCASE_API int nncase_compiler_compile_options_set_dump_level(
    clr_object_handle compile_options, int32_t dump_level);
}
