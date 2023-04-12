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
#include "../compiler_defs.h"
#include <functional>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace nncase {
#define try_(x)                                                                \
    {                                                                          \
        auto v = (x);                                                          \
        if (!v.is_ok())                                                        \
            return nncase::err(std::move(v.unwrap_err()));                     \
    }

#define try_var(name, x)                                                       \
    typename decltype((x))::value_type name;                                   \
    {                                                                          \
        auto v = (x);                                                          \
        if (v.is_ok())                                                         \
            name = std::move(v.unwrap());                                      \
        else                                                                   \
            return nncase::err(std::move(v.unwrap_err()));                     \
    }

#define try_var_err(name, x, e)                                                \
    typename decltype((x))::value_type name;                                   \
    {                                                                          \
        auto v = (x);                                                          \
        if (v.is_ok()) {                                                       \
            name = std::move(v.unwrap());                                      \
        } else {                                                               \
            e = nncase::err(std::move(v.unwrap_err()));                        \
            return;                                                            \
        }                                                                      \
    }

#define try_set(name, x)                                                       \
    {                                                                          \
        auto v = (x);                                                          \
        if (v.is_ok())                                                         \
            name = std::move(v.unwrap());                                      \
        else                                                                   \
            return nncase::err(std::move(v.unwrap_err()));                     \
    }

[[noreturn]] inline void fail_fast(const char *message) {
    fprintf(stderr, "terminate:%s\n", message);
    // auto exit for pld
    fprintf(stderr, "}");
    std::terminate();
}

template <class T> class NNCASE_NODISCARD result;

namespace detail {
enum class result_type { ok, err };

struct ok_t {};
NNCASE_INLINE_VAR ok_t constexpr ok_v = {};

template <class T> NNCASE_INLINE_VAR bool constexpr is_result_v = false;
template <class T>
NNCASE_INLINE_VAR bool constexpr is_result_v<result<T>> = true;

template <class T, class U, class Func> class map_call_impl {
    result<U> operator()(Func &&func, T &value) noexcept {
        return ok(func(value));
    }
};

template <class T, class Func> struct map_traits {
    using U = invoke_result_t<Func, T>;
    static_assert(
        !is_result_v<U>,
        "Cannot map a callback returning result, use and_then instead");
    using result_t = result<U>;

    result<U> operator()(Func &&func, T &value) noexcept {
        return map_call_impl<T, U, Func>()(std::forward<Func>(func), value);
    }
};

template <class Func> struct map_traits<void, Func> {
    using U = invoke_result_t<Func>;
    static_assert(
        !is_result_v<U>,
        "Cannot map a callback returning result, use and_then instead");
    using result_t = result<U>;

    result<U> operator()(Func &&func) noexcept {
        return map_call_impl<void, U, Func>()(std::forward<Func>(func));
    }
};

template <class T, class Func> struct map_err_traits {
    using U = invoke_result_t<Func, std::error_condition>;
    static_assert(
        !is_result_v<U>,
        "Cannot map a callback returning result, use and_then instead");

    result<U> operator()(Func &&func, std::error_condition &value) noexcept {
        return err(func(value));
    }
};

template <class T, class Func> struct map_err_traits;

template <class T, class Func> struct and_then_traits {
    using result_t = invoke_result_t<Func, T>;
    using traits_t = typename result_t::traits;
    using U = typename traits_t::ok_type;
    static_assert(
        is_result_v<result_t>,
        "Cannot then a callback not returning result, use map instead");

    result_t operator()(Func &&func, T &value) noexcept { return func(value); }
};

template <class Func> struct and_then_traits<void, Func> {
    using result_t = invoke_result_t<Func>;
    using traits_t = typename result_t::traits;
    using U = typename traits_t::ok_type;
    static_assert(
        is_result_v<result_t>,
        "Cannot then a callback not returning result, use map instead");

    result_t operator()(Func &&func) noexcept { return func(); }
};
} // namespace detail

template <class T> class NNCASE_NODISCARD result {
  public:
    static_assert(!detail::is_result_v<T>, "Cannot use nested result");

    using value_type = T;

    template <class... Args>
    result(detail::ok_t, Args... args)
        : type_(detail::result_type::ok), ok_(std::forward<Args>(args)...) {}

    result(std::error_condition err) noexcept
        : type_(detail::result_type::err), err_(std::move(err)) {}

    result(const result &other) : type_(other.type_) {
        if (type_ == detail::result_type::ok)
            new (&ok_) T(other.ok_);
        else
            new (&err_) std::error_condition(other.err_);
    }

    result(result &&other) : type_(other.type_) {
        if (type_ == detail::result_type::ok)
            new (&ok_) T(std::move(other.ok_));
        else
            new (&err_) std::error_condition(std::move(other.err_));
    }

    template <class U, class = std::enable_if_t<std::is_convertible_v<U, T>>>
    result(result<U> &&other) : type_(other.type_) {
        if (type_ == detail::result_type::ok)
            new (&ok_) T(std::move(other.ok_));
        else
            new (&err_) std::error_condition(std::move(other.err_));
    }

    ~result() { destroy(); }

    result &operator=(const result &other) noexcept {
        destroy();
        type_ = other.type_;
        if (type_ == detail::result_type::ok)
            new (&ok_) T(other.ok_);
        else
            new (&err_) std::error_condition(other.err_);
        return *this;
    }

    result &operator=(result &&other) noexcept {
        destroy();
        type_ = other.type_;
        if (type_ == detail::result_type::ok)
            new (&ok_) T(std::move(other.ok_));
        else
            new (&err_) std::error_condition(std::move(other.err_));
        return *this;
    }

    constexpr bool is_ok() const noexcept {
        return type_ == detail::result_type::ok;
    }

    constexpr bool is_err() const noexcept {
        return type_ == detail::result_type::err;
    }

    constexpr T &unwrap() &noexcept {
        if (is_ok())
            return ok_;
        else
            std::terminate();
    }

    constexpr T &&unwrap() &&noexcept {
        if (is_ok())
            return std::move(ok_);
        else
            std::terminate();
    }

    constexpr T &unwrap_or_throw() & {
        if (is_ok())
            return ok_;
        else
            throw std::runtime_error(unwrap_err().message());
    }

    constexpr T &&unwrap_or_throw() && {
        if (is_ok())
            return std::move(ok_);
        else
            throw std::runtime_error(unwrap_err().message());
    }

    constexpr std::error_condition &unwrap_err() noexcept {
        if (is_ok())
            std::terminate();
        else
            return err_;
    }

    constexpr T &expect(gsl::cstring_span message) &noexcept {
        if (is_ok())
            return ok_;
        else {
            fail_fast(message.data());
        }
    }

    constexpr T &&expect(gsl::cstring_span message) &&noexcept {
        if (is_ok())
            return std::move(ok_);
        else {
            fail_fast(message.data());
        }
    }

    template <class Func, class Traits = detail::map_traits<T, Func>>
    constexpr typename Traits::result_t &&map(Func &&func) &&noexcept {
        if (is_ok())
            return Traits()(std::forward<Func>(func), std::move(ok_));
        else
            return std::move(*this);
    }

    template <class Func, class Traits = detail::map_err_traits<T, Func>>
    constexpr typename Traits::result_t &&map_err(Func &&func) &&noexcept {
        if (is_ok())
            return std::move(*this);
        else
            return Traits()(std::forward<Func>(func), err_);
    }

    template <class Func, class Traits = detail::and_then_traits<T, Func>>
    constexpr typename Traits::result_t &&and_then(Func &&func) &&noexcept {
        if (is_ok())
            return Traits()(std::forward<Func>(func), ok_);
        else
            return std::move(*this);
    }

  private:
    void destroy() {
        if (is_ok())
            std::destroy_at(&ok_);
        else
            std::destroy_at(&err_);
    }

  private:
    template <class U> friend class result;

    detail::result_type type_;
    union {
        T ok_;
        std::error_condition err_;
    };
};

template <> class NNCASE_NODISCARD result<void> {
  public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
    result() noexcept : err_(0, *(std::error_category *)nullptr) {}
#pragma GCC diagnostic pop

    result(std::error_condition err) noexcept : err_(std::move(err)) {}

    bool is_ok() const noexcept { return !is_err(); }
    bool is_err() const noexcept { return (bool)err_; }

    void unwrap() noexcept {
        if (is_err())
            std::terminate();
    }

    void unwrap_or_throw() {
        if (is_err())
            throw std::runtime_error(unwrap_err().message());
    }

    std::error_condition &unwrap_err() noexcept {
        if (is_ok())
            std::terminate();
        else
            return err_;
    }

    void expect(gsl::cstring_span message) noexcept {
        if (is_err())
            fail_fast(message.data());
    }

    template <class Func, class Traits = detail::map_traits<void, Func>>
    typename Traits::result_t &&map(Func &&func) &&noexcept {
        if (is_ok())
            return Traits()(std::forward<Func>(func));
        else
            return std::move(*this);
    }

    template <class Func, class Traits = detail::map_err_traits<void, Func>>
    typename Traits::result_t &&map_err(Func &&func) &&noexcept {
        if (is_ok())
            return std::move(*this);
        else
            return Traits()(std::forward<Func>(func), err_);
    }

    template <class Func, class Traits = detail::and_then_traits<void, Func>>
    typename Traits::result_t &&and_then(Func &&func) &&noexcept {
        if (is_ok())
            return Traits()(std::forward<Func>(func));
        else
            return std::move(*this);
    }

  private:
    std::error_condition err_;
};

inline result<void> ok() { return {}; }

template <class T, class... Args> constexpr result<T> ok(Args &&...args) {
    return {detail::ok_v, std::forward<Args>(args)...};
}

template <class T> constexpr result<std::decay_t<T>> ok(T &&value) {
    return {detail::ok_v, std::forward<T>(value)};
}

inline std::error_condition err(std::error_condition value) noexcept {
    return value;
}

template <class ErrCode, class = std::enable_if_t<
                             std::is_error_condition_enum<ErrCode>::value>>
std::error_condition err(ErrCode value) {
    return err(std::error_condition(value));
}

namespace detail {
template <class T, class Func> class map_call_impl<T, void, Func> {
    result<void> operator()(Func &&func, T &value) noexcept {
        func(value);
        return ok();
    }
};

template <class Func> class map_call_impl<void, void, Func> {
    result<void> operator()(Func &&func) noexcept {
        func();
        return ok();
    }
};
} // namespace detail
} // namespace nncase
