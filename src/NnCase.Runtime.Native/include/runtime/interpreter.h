#pragma once
#include "model.h"
#include <memory>
#include <xtl/xspan.hpp>
#include "op_utility.h"

namespace nncase
{
namespace runtime
{
    typedef void (*run_callback_t)(void *userdata);

    class interpreter
    {
    public:
        bool try_load_model(const uint8_t *buffer);

        size_t inputs_size() const noexcept { return model_header_->inputs; }
        size_t outputs_size() const noexcept { return model_header_->outputs; }

        const runtime_shape_t &input_shape_at(size_t index) const noexcept { return input_shapes_.at(index); }
        const memory_range &input_at(size_t index) const noexcept { return inputs_[index]; }
        const memory_range &output_at(size_t index) const noexcept { return outputs_[index]; }

        template <class T>
        xtl::span<T> memory_at(const memory_range &range) const noexcept
        {
            auto span = memory_at(range);
            return { reinterpret_cast<T>(span.data()), span.size() / get_bytes(range.datatype) };
        }

        void run(run_callback_t callback);

    private:
        xtl::span<uint8_t> memory_at(const memory_range &range) const noexcept;

    private:
        const model_header *model_header_;
        std::unique_ptr<uint8_t[]> main_mem_;
        xtl::span<const memory_range> inputs_;
        xtl::span<const memory_range> outputs_;
        xtl::span<const runtime_shape_t> input_shapes_;
    };
}
}
