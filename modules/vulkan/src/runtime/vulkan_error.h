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
#include <nncase/runtime/vulkan/runtime_module.h>
#include <system_error>
#include <vulkan/vulkan.hpp>

namespace vk
{
NNCASE_API const std::error_category &vulkan_category() noexcept;
NNCASE_API std::error_condition make_error_condition(vk::Result code);
}

namespace std
{
template <>
struct is_error_condition_enum<vk::Result> : true_type
{
};
}

namespace vk
{
template <class T>
nncase::result<T> to_result(vk::ResultValue<T> &&value) noexcept
{
    if (value.result == vk::Result::eSuccess)
        return nncase::ok(std::move(value.value));
    else
        return nncase::err(value.result);
}
}
