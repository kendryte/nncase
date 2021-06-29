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
#include <memory>
#include <nncase/runtime/compiler_defs.h>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API allocation_state
{
public:
    virtual ~allocation_state();
};

class NNCASE_API host_allocator
{
public:
    virtual ~host_allocator();
    virtual gsl::span<gsl::byte> allocate(allocation_state &state, size_t bytes) = 0;
};

END_NS_NNCASE_RUNTIME
