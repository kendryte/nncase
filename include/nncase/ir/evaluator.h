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
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/schedule/scheduler.h>
#include <unordered_map>
#include <xtensor/xadapt.hpp>

namespace nncase::ir
{
template <class T = std::byte>
struct eval_result
{
    std::span<T> span;
    shape_t shape;
    shape_t strides;

    auto view()
    {
        return xt::adapt(span.data(), span.size(), xt::no_ownership(), shape, strides);
    }
};

class NNCASE_API evaluator
{
public:
    evaluator(const schedule::schedule_result &sched);
    evaluator(evaluator &) = delete;

    eval_result<> memory_at(const output_connector &conn);

    template <class T>
    eval_result<T> memory_at(const output_connector &conn)
    {
        auto result = memory_at(conn);
        return {
            { reinterpret_cast<T *>(result.span.data()), result.span.size_bytes() / sizeof(T) },
            ir::convert_shape_type(result.shape, dt_uint8, to_datatype<T>()),
            ir::convert_strides_type(result.strides, dt_uint8, to_datatype<T>())
        };
    }

    template <class T>
    eval_result<T> memory_at(const input_connector &conn)
    {
        return memory_at<T>(*conn.connection());
    }

    template <class T>
    eval_result<T> input_at(size_t index)
    {
        return memory_at<T>(*inputs_[index]);
    }

    template <class T>
    eval_result<T> output_at(size_t index)
    {
        return memory_at<T>(*outputs_[index]);
    }

    void evaluate();

private:
    const schedule::schedule_result &sched_;
    std::unordered_map<memory_location_t, std::unique_ptr<std::byte[]>> memory_pools_;

    std::vector<output_connector *> inputs_;
    std::vector<input_connector *> outputs_;
};

NNCASE_API void register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, evaluator &)> evaluator);
}
