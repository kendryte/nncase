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
#include "object_kind.h"
#include "runtime/datatypes.h"
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace nncase {
#define DEFINE_OBJECT_KIND(base_t, kind_)                                      \
  public:                                                                      \
    static constexpr object_kind kind() noexcept { return kind_; }             \
    const object_kind &runtime_kind() const noexcept override {                \
        return kind_;                                                          \
    }                                                                          \
                                                                               \
  protected:                                                                   \
    bool is_a(const object_kind &kind) const noexcept override {               \
        return kind == kind_ || base_t::is_a(kind);                            \
    }

template <class T> class object_t;
template <class T> concept Object = requires { typename T::node_type; };

class NNCASE_API object_node {
  public:
    constexpr object_node() noexcept = default;
    object_node(const object_node &) = delete;
    object_node &operator=(const object_node &) = delete;
    virtual ~object_node();

    /** @brief Get the kind of the object */
    virtual const object_kind &runtime_kind() const noexcept = 0;

  protected:
    template <class T> friend class object_t;

    /** @brief Is the object an instance of specific kind */
    virtual bool is_a(const object_kind &kind) const noexcept;
};

template <class T> class object_t {
  public:
    using node_type = T;

    static inline constexpr bool is_default_constructible_v =
        std::is_default_constructible_v<T> && !std::is_abstract_v<T>;

    object_t() : object_(default_construct()) {
        static_assert(is_default_constructible_v,
                      "Use nullptr to construct an empty object.");
    }

    /** @brief Construct an empty object */
    constexpr object_t(std::nullptr_t) noexcept : object_(nullptr) {}

    explicit object_t(std::shared_ptr<T> node) noexcept
        : object_(std::move(node)) {}

    template <Object U, class = std::enable_if_t<
                            std::is_base_of_v<T, typename U::node_type>>>
    object_t(U &&other) noexcept : object_(std::move(other.object_)) {}

    template <Object U, class = std::enable_if_t<
                            std::is_base_of_v<T, typename U::node_type>>>
    object_t(const U &other) noexcept : object_(other.object_) {}

    template <class... TArgs>
    object_t(std::in_place_t, TArgs &&...args)
        : object_(std::make_shared<T>(std::forward<TArgs>(args)...)) {}

    /** @brief Get the managed pointer to the object */
    T *get() const noexcept { return static_cast<T *>(object_.get()); }
    T *operator->() const noexcept { return get(); }

    bool empty() const noexcept { return object_.get() == nullptr; }

    /** @brief Is the object an instance of specific type */
    bool is_a(const object_kind &kind) const noexcept {
        return object_ && static_cast<object_node *>(object_.get())->is_a(kind);
    }

    /** @brief Is the object an instance of specific type */
    template <Object U> bool is_a() const noexcept {
        return is_a(U::node_type::kind());
    }

    template <Object U> std::optional<U> as() const noexcept {
        if (is_a<U>()) {
            return std::make_optional(
                U(std::static_pointer_cast<typename U::node_type>(object_)));
        } else {
            return std::nullopt;
        }
    }

  private:
    std::shared_ptr<T> default_construct() {
        if constexpr (is_default_constructible_v)
            return std::make_shared<T>();
        else
            return nullptr;
    }

  private:
    template <class U> friend class object_t;

    std::shared_ptr<T> object_;
};

using object = object_t<object_node>;
} // namespace nncase
