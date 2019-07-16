#pragma once
#include "model.h"
#include "op_utility.h"
#include <chrono>
#include <memory>
#include <optional>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace runtime
{
    class interpreter;
    typedef void (*run_callback_t)(void *userdata);
    typedef void (*error_callback_t)(const char *err, void *userdata);
    typedef void (*node_profile_callback_t)(runtime_opcode op, std::chrono::nanoseconds duration, void *userdata);
    typedef void (interpreter::*interpreter_step_t)();

    class interpreter
    {
        using clock_t = std::chrono::system_clock;

    public:
        bool try_load_model(const uint8_t *buffer);

        size_t inputs_size() const noexcept { return model_header_->inputs; }
        size_t outputs_size() const noexcept { return model_header_->outputs; }
        size_t nodes_size() const noexcept { return model_header_->nodes; }

        const runtime_shape_t &input_shape_at(size_t index) const noexcept { return input_shapes_.at(index); }
        const memory_range &input_at(size_t index) const noexcept { return inputs_[index]; }
        const memory_range &output_at(size_t index) const noexcept { return outputs_[index]; }

        template <class T>
        xtl::span<T> memory_at(const memory_range &range) const noexcept
        {
            auto span = memory_at(range);
            return { reinterpret_cast<T *>(span.data()), span.size() / sizeof(T) };
        }

        std::chrono::nanoseconds total_duration() const noexcept { return total_duration_; }

        void run(run_callback_t callback, error_callback_t on_error, node_profile_callback_t node_profile, void *userdata);

    private:
        void step();

    private:
        xtl::span<uint8_t> memory_at(const memory_range &range) const noexcept;

    private:
        const model_header *model_header_;
        std::unique_ptr<uint8_t[]> main_mem_;
        xtl::span<const memory_range> inputs_;
        xtl::span<const memory_range> outputs_;
        xtl::span<const runtime_shape_t> input_shapes_;
        xtl::span<const node_header> node_headers_;
        xtl::span<const uint8_t> constants_;
        const uint8_t *node_body_start_;
        error_callback_t on_error_;
        run_callback_t run_callback_;
        node_profile_callback_t node_profile_;
        void *userdata_;
        size_t cnt_node_;
        const uint8_t *cnt_node_body_;
        std::chrono::nanoseconds total_duration_;
        std::optional<clock_t::time_point> last_time_;
        runtime_opcode last_op_;
    };
}
}
