#pragma once
#include <xtl/xspan.hpp>

namespace nncase
{
namespace runtime
{
    class span_reader
    {
    public:
        span_reader(xtl::span<const uint8_t> span)
            : span_(span)
        {
        }

        bool empty() const noexcept { return span_.empty(); }

        template <class T>
        T read()
        {
            auto value = *reinterpret_cast<const T *>(span_.data());
            advance(sizeof(T));
            return value;
        }

        template <class T>
        void read(T &value)
        {
            value = *reinterpret_cast<const T *>(span_.data());
            advance(sizeof(T));
        }

        template <class T>
        void read_span(xtl::span<const T> &span, size_t size)
        {
            span = { reinterpret_cast<const T *>(span_.data()), size };
            advance(sizeof(T) * size);
        }

        template <class T, ptrdiff_t N>
        void read_span(xtl::span<const T, N> &span)
        {
            span = { reinterpret_cast<const T *>(span_.data()), N };
            advance(sizeof(T) * N);
        }

        template <class T>
        const T *peek() const noexcept
        {
            return reinterpret_cast<const T *>(span_.data());
        }

        template <class T>
        void get_array(const T *&value, size_t size)
        {
            value = peek<T>();
            advance(size * sizeof(T));
        }

        template <class T>
        void get_ref(const T *&value)
        {
            value = peek<T>();
            advance(sizeof(T));
        }

        void skip(size_t count)
        {
            advance(count);
        }

    private:
        void advance(size_t count)
        {
            span_ = span_.subspan(count);
        }

    private:
        xtl::span<const uint8_t> span_;
    };
}
}
