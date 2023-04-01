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
#include <atomic>
#include <memory>
#include <nncase/api.h>
#include <nncase/runtime/result.h>
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

class NNCASE_API object_node {
  public:
    object_node() noexcept;
    object_node(const object_node &) = delete;
    object_node &operator=(const object_node &) = delete;
    virtual ~object_node() = default;

    /** @brief Get the kind of the object */
    virtual const object_kind &runtime_kind() const noexcept = 0;

  protected:
    template <class T> friend class object_t;

    /** @brief Is the object an instance of specific kind */
    virtual bool is_a(const object_kind &kind) const noexcept;

    /** @brief Is the object equal to another instance */
    virtual bool equals(const object_node &other) const noexcept;

  private:
    uint32_t add_ref() const noexcept {
        return ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    uint32_t release() const noexcept {
        assert(ref_count_);
        auto count = ref_count_.fetch_sub(1, std::memory_order_acq_rel);
        if (count == 1) {
            delete this;
        }
        return count;
    }

    template <class T> friend class object_t;
    friend int ::nncase_object_add_ref(nncase::object_node *node);
    friend int ::nncase_object_release(nncase::object_node *node);

  private:
    mutable std::atomic<uint32_t> ref_count_;
};

template <class T> class object_t {
  public:
    using node_type = T;

    /** @brief Construct an empty object */
    constexpr object_t(std::nullptr_t = nullptr) noexcept : object_(nullptr) {}
    ~object_t() { release(); }

    object_t(T *node) noexcept : object_(node) { add_ref(); }
    object_t(std::in_place_t, T *node) noexcept : object_(node) {}

    object_t(object_t &&other) noexcept : object_(other.object_) {
        other.object_ = nullptr;
    }

    object_t(const object_t &other) noexcept : object_(other.object_) {
        add_ref();
    }

    template <class U,
              class = std::enable_if_t<std::is_convertible_v<U *, T *>>>
    object_t(object_t<U> &&other) noexcept : object_(other.object_) {
        other.object_ = nullptr;
    }

    template <class U,
              class = std::enable_if_t<std::is_convertible_v<U *, T *>>>
    object_t(const object_t<U> &other) noexcept : object_(other.object_) {
        add_ref();
    }

    template <class... TArgs>
    object_t(std::in_place_t, TArgs &&...args)
        : object_(new T(std::forward<TArgs>(args)...)) {}

    /** @brief Get the managed pointer to the object */
    T *get() const noexcept { return object_; }
    T *operator->() const noexcept { return get(); }

    bool empty() const noexcept { return !object_; }

    object_t value_or(object_t &&other) const noexcept {
        if (!empty())
            return *this;
        return std::move(other);
    }

    /** @brief Is the object an instance of specific type */
    bool is_a(const object_kind &kind) const noexcept {
        return object_ && static_cast<object_node *>(object_)->is_a(kind);
    }

    /** @brief Is the object an instance of specific type */
    template <class U> bool is_a() const noexcept {
        return is_a(U::node_type::kind());
    }

    template <class U> result<U> as() const noexcept {
        if (is_a<U>()) {
            return ok(U(static_cast<typename U::node_type *>(object_)));
        } else {
            return err(std::errc::invalid_argument);
        }
    }

    /** @brief Is the object equal to another instance */
    template <class U> bool equals(const U &other) const noexcept {
        if (get() == other.get())
            return true;
        else if (!empty() && !other.empty())
            return object_->equals(*other.get());
        return false;
    }

    object_t &operator=(object_t &&other) noexcept {
        if (this != &other) {
            release();
            object_ = other.object_;
            other.object_ = nullptr;
        }
        return *this;
    }

    object_t &operator=(const object_t &other) noexcept {
        if (this != &other) {
            release();
            object_ = other.object_;
            add_ref();
        }
        return *this;
    }

    T *detach() noexcept {
        auto obj = object_;
        object_ = nullptr;
        return obj;
    }

    T **release_and_addressof() noexcept {
        release();
        return &object_;
    }

    void dangerous_add_ref() noexcept { return add_ref(); }

  private:
    void add_ref() noexcept {
        if (object_) {
            object_->add_ref();
        }
    }

    void release() noexcept {
        auto obj = object_;
        if (obj) {
            obj->release();
            object_ = nullptr;
        }
    }

  private:
    template <class U> friend class object_t;

    T *object_;
};

using object = object_t<object_node>;
} // namespace nncase
