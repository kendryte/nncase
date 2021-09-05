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
#include "runtime/datatypes.h"
#include <memory>
#include <type_traits>
#include <utility>

namespace nncase
{
template <class T>
class object_ref
{
public:
    static inline constexpr bool is_default_constructible_v = std::is_default_constructible_v<T> && !std::is_abstract_v<T>;

    object_ref()
        : object_(default_construct())
    {
        static_assert(is_default_constructible_v, "Use nullptr to construct an empty object.");
    }

    /** @brief Construct an empty object **/
    constexpr object_ref(std::nullptr_t) noexcept
        : object_(nullptr)
    {
    }

    object_ref(std::shared_ptr<T> node) noexcept
        : object_(std::move(node))
    {
    }

    template <class = std::enable_if_t<is_default_constructible_v>, class... TArgs>
    object_ref(std::in_place_t, TArgs &&...args)
        : object_(std::make_shared<T>(std::forward<TArgs>(args)...))
    {
    }

    /** @brief Get the managed pointer to the object **/
    T *get() const noexcept { return object_.get(); }
    T *operator->() const noexcept { return get(); }

private:
    std::shared_ptr<T> default_construct()
    {
        if constexpr (is_default_constructible_v)
            return std::make_shared<T>();
        else
            return nullptr;
    }

private:
    std::shared_ptr<T> object_;
};
}
