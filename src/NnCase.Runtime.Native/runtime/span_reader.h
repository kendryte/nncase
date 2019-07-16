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

        template <class T>
        const T *get() const noexcept
        {
            return reinterpret_cast<const T *>(span_.data());
        }

        template <class T>
        void get_array(const T *&value, size_t size)
        {
            value = get<T>();
            advance(size * sizeof(T));
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
