#pragma once
#include <iostream>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace runtime
{
    class binary_writer
    {
    public:
        binary_writer(std::ostream &stream)
            : stream_(stream)
        {
        }

        template <class T>
        void write(T &&value)
        {
            stream_.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }

        template <class T>
        void write_array(xtl::span<const T> value)
        {
            stream_.write(reinterpret_cast<const char *>(value.data()), value.size_bytes());
        }

        std::streampos position() const
        {
            return stream_.tellp();
        }

        void position(std::streampos pos)
        {
            stream_.seekp(pos);
        }

        void align_position(size_t alignment)
        {
            auto pos = position();
            auto rem = pos % alignment;
            if (rem != 0)
                position(pos + std::streamoff(alignment - rem));
        }

    private:
        std::ostream &stream_;
    };
}
}
