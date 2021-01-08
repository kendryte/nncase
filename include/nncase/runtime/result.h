/* Copyright 2020 Canaan Inc.
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
#include "compiler_defs.h"
#include <functional>
#include <mpark/variant.hpp>
#include <system_error>
#include <type_traits>

namespace nncase
{
#define try_(x)                                            \
    {                                                      \
        auto v = (x);                                      \
        if (!v.is_ok())                                    \
            return nncase::err(std::move(v.unwrap_err())); \
    }

#define try_var(name, x)                                   \
    typename decltype((x))::traits::ok_type name;          \
    {                                                      \
        auto v = (x);                                      \
        if (v.is_ok())                                     \
            name = std::move(v.unwrap());                  \
        else                                               \
            return nncase::err(std::move(v.unwrap_err())); \
    }

#define try_set(name, x)                                   \
    {                                                      \
        auto v = (x);                                      \
        if (v.is_ok())                                     \
            name = std::move(v.unwrap());                  \
        else                                               \
            return nncase::err(std::move(v.unwrap_err())); \
    }

template <class T>
struct Ok
{
    constexpr Ok(T &&value)
        : value(std::move(value)) { }

    constexpr Ok(const T &value)
        : value(value) { }

    template <class... Args>
    constexpr explicit Ok(mpark::in_place_t, Args &&... args)
        : value(std::forward<Args>(args)...) { }

    T value;
};

template <>
struct Ok<void>
{
};

struct Err
{
    template <class ErrCode, class = std::enable_if_t<std::is_error_condition_enum_v<ErrCode>>>
    constexpr Err(ErrCode value)
        : err(value) { }

    Err(std::error_condition err)
        : err(std::move(err)) { }

    std::error_condition err;
};

inline constexpr Ok<void> ok()
{
    return {};
}

template <class T, class... Args>
constexpr Ok<T> ok(Args &&... args)
{
    return Ok<T>(mpark::in_place, std::forward<Args>(args)...);
}

template <class T>
constexpr Ok<std::decay_t<T>> ok(T &&value)
{
    return Ok<std::decay_t<T>>(std::forward<T>(value));
}

template <class ErrCode, class = std::enable_if_t<std::is_error_condition_enum_v<ErrCode>>>
constexpr Err err(ErrCode value)
{
    return Err(value);
}

Err err(std::error_condition value)
{
    return Err(std::move(value));
}

template <class T>
class NNCASE_NODISCARD result;

namespace details
{
    template <class T>
    NNCASE_INLINE_VAR bool constexpr is_result_v = false;
    template <class T>
    NNCASE_INLINE_VAR bool constexpr is_result_v<result<T>> = true;

    template <class T>
    struct result_traits
    {
        static_assert(!is_result_v<T>, "Cannot use nested result");

        using ok_type = T;
    };

    template <class T, class U, class Func>
    class map_call_impl
    {
        result<U> operator()(Func &&func, Ok<T> &value) noexcept
        {
            return ok(func(value.value));
        }
    };

    template <class T, class Func>
    struct map_traits;

    template <class U, class Func>
    class map_call_void_impl
    {
        result<U> operator()(Func &&func) noexcept
        {
            return ok(func());
        }
    };

    template <class Func>
    struct map_traits<void, Func>
    {
        using U = invoke_result_t<Func>;
        static_assert(!is_result_v<U>, "Cannot map a callback returning result, use and_then instead");

        result<U> operator()(Func &&func, Ok<void> &value) noexcept
        {
            return map_call_void_impl<U, Func>()(std::forward<Func>(func));
        }
    };

    template <class T, class Func>
    struct map_err_traits;

    template <class T, class Func>
    struct and_then_traits
    {
        using result_t = invoke_result_t<Func, T>;
        using traits_t = typename result_t::traits;
        using U = typename traits_t::ok_type;
        static_assert(is_result_v<result_t>, "Cannot then a callback not returning result, use map instead");

        result_t operator()(Func &&func, Ok<T> &value) noexcept
        {
            return func(value.value);
        }
    };

    template <class Func>
    struct and_then_traits<void, Func>
    {
        using result_t = invoke_result_t<Func>;
        using traits_t = typename result_t::traits;
        using U = typename traits_t::ok_type;
        static_assert(is_result_v<result_t>, "Cannot then a callback not returning result, use map instead");

        result_t operator()(Func &&func, Ok<void> &value) noexcept
        {
            return func();
        }
    };

    template <class T>
    struct unwrap_impl
    {
        T &operator()(Ok<T> &value) noexcept
        {
            return value.value;
        }
    };

    template <>
    struct unwrap_impl<void>
    {
        void operator()(Ok<void> &value) noexcept
        {
        }
    };
}

template <class T>
class NNCASE_NODISCARD result
{
public:
    using traits = details::result_traits<T>;

    constexpr result(Ok<T> value)
        : ok_or_err_(std::move(value)) { }

    constexpr result(Err err)
        : ok_or_err_(std::move(err)) { }

    constexpr bool is_ok() const noexcept { return ok_or_err_.index() == 0; }
    constexpr bool is_err() const noexcept { return ok_or_err_.index() == 1; }

    constexpr decltype(auto) unwrap() noexcept
    {
        if (is_ok())
            return details::unwrap_impl<T>()(value());
        else
            std::terminate();
    }

    constexpr std::error_condition &unwrap_err() noexcept
    {
        if (is_ok())
            std::terminate();
        else
            return err().err;
    }

    constexpr auto expect(gsl::cstring_span message) noexcept
    {
        if (is_ok())
        {
            if constexpr (std::is_same_v<T, void>)
                return;
            else
                return std::ref(value().value);
        }
        else
        {
            std::terminate();
        }
    }

    template <class Func, class Traits = details::map_traits<T, Func>>
    constexpr typename Traits::result_t map(Func &&func) noexcept
    {
        if (is_ok())
            return Traits()(std::forward<Func>(func), value());
        else
            return err();
    }

    template <class Func, class Traits = details::map_err_traits<T, Func>>
    constexpr typename Traits::result_t map_err(Func &&func) noexcept
    {
        if (is_ok())
            return value();
        else
            return Traits()(std::forward<Func>(func), err());
    }

    template <class Func, class Traits = details::and_then_traits<T, Func>>
    constexpr typename Traits::result_t and_then(Func &&func) noexcept
    {
        if (is_ok())
            return Traits()(std::forward<Func>(func), value());
        else
            return err();
    }

private:
    constexpr Ok<T> &value() noexcept { return mpark::get<Ok<T>>(ok_or_err_); }
    constexpr Err &err() noexcept { return mpark::get<Err>(ok_or_err_); }

private:
    mpark::variant<Ok<T>, Err> ok_or_err_;
};

namespace details
{
    template <class T, class Func>
    struct map_traits
    {
        using U = invoke_result_t<Func, T>;
        static_assert(!is_result_v<U>, "Cannot map a callback returning result, use and_then instead");

        result<U> operator()(Func &&func, Ok<T> &value) noexcept
        {
            return map_call_impl<T, U, Func>()(std::forward<Func>(func), value);
        }
    };

    template <class T, class Func>
    struct map_err_traits
    {
        using U = invoke_result_t<Func, Err>;
        static_assert(!is_result_v<U>, "Cannot map a callback returning result, use and_then instead");

        result<U> operator()(Func &&func, Err &value) noexcept
        {
            return err(func(value.err));
        }
    };

    template <class T, class Func>
    class map_call_impl<T, void, Func>
    {
        result<void> operator()(Func &&func, Ok<T> &value) noexcept
        {
            func(value.value);
            return ok();
        }
    };

    template <class Func>
    class map_call_void_impl<void, Func>
    {
        result<void> operator()(Func &&func) noexcept
        {
            func();
            return ok();
        }
    };
}
}
