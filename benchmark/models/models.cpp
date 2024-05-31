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
#include "models.h"
#include <system_error>

using namespace nncase;

#if _MSC_VER
#include "resource.h"
#include <Windows.h>

namespace
{
std::span<const std::byte> get_model_impl(const std::string &name, size_t id)
{
    auto hres = FindResourceW(NULL, MAKEINTRESOURCEW(id), L"Binary");
    if (!hres)
        return {};
    auto size = SizeofResource(NULL, hres);
    auto hmem = LoadResource(NULL, hres);
    if (!hmem)
        return {};
    auto res_data = LockResource(hmem);
    return { reinterpret_cast<const std::byte *>(res_data), (size_t)size };
}
}

#define GET_MODEL_IMPL(model) \
    if (name == #model)       \
    return get_model_impl(name, IDR_cpu_##model)

std::span<const std::byte> nncase::get_model(const std::string &name)
{
    GET_MODEL_IMPL(mnist);
    GET_MODEL_IMPL(mobilenet_v2);
    return {};
}
#else
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <nncase/runtime/incbin.h>
INCBIN(mnist, "cpu/mnist.kmodel");
INCBIN(mobilenet_v2, "cpu/mobilenet_v2.kmodel");

#define GET_MODEL_IMPL(model) \
    if (name == #model)       \
        return { reinterpret_cast<const std::byte *>(g##model##_data), g##model##_size }

std::span<const std::byte> nncase::get_model(const std::string &name)
{
    GET_MODEL_IMPL(mnist);
    GET_MODEL_IMPL(mobilenet_v2);
    return {};
}

#endif
