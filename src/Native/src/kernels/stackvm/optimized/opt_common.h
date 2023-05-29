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
#include <cstring>

inline void *opt_memcpy(void *dst, const void *src, size_t n) {
#if __riscv_vector
    return memcpy_vec(out_ptr, in_ptr, copy_size);
#else
    return memcpy(dst, src, n);
#endif
}