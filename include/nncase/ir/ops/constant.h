/* Copyright 2019-2020 Canaan Inc.
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
#include "../debug.h"
#include "../node.h"
#include <vector>

namespace nncase::ir
{
class constant : public node
{
public:
    DEFINE_NODE_OPCODE(op_constant);

    output_connector &output() { return output_at(0); }
    const output_connector &output() const { return output_at(0); }

    std::span<const std::byte> data() const noexcept { return data_; }
    datatype_t data_type() { return datatype_; }

    template <class TShape>
    constant(datatype_t type, TShape &&shape, std::span<const std::byte> data)
        : constant(type, std::forward<TShape>(shape), data.begin(), data.end())
    {
    }

    template <class TShape, class T>
    constant(datatype_t type, TShape &&shape, std::span<const T> data)
        : constant(type, std::forward<TShape>(shape), std::as_bytes(data))
    {
    }

    template <class TShape, class T>
    constant(datatype_t type, TShape &&shape, std::span<T> data)
        : constant(type, std::forward<TShape>(shape), std::as_bytes(data))
    {
    }

    template <class TShape, class T>
    constant(datatype_t type, TShape &&shape, const std::vector<T> &data)
        : constant(type, std::forward<TShape>(shape), std::as_bytes(std::span<const T>(data)))
    {
    }

    template <class TScalar>
    constant(TScalar scalar)
        : constant(to_datatype<TScalar>(), shape_t { 1 }, std::span<const TScalar>(&scalar, 1))
    {
    }

    template <class TShape, class... TDataArgs>
    constant(datatype_t type, TShape &&shape, TDataArgs... data_args)
        : data_(std::forward<TDataArgs>(data_args)...), datatype_(type)
    {
        add_output("output", type, std::forward<TShape>(shape));
    }

    std::string to_string() const
    {
        auto shape = this->output().shape();
        auto dtype = this->output().type();
        auto total_size = 1;
        for (auto i : shape)
        {
            total_size *= i;
        }

        if (total_size == 1)
        {
            switch (dtype)
            {
            case dt_int8:
                return std::to_string(*(to_cpp_type_t<dt_int8> *)data_.data());
            case dt_uint8:
                return std::to_string(*(to_cpp_type_t<dt_uint8> *)data_.data());
#define DT_TO_STRING_CASE(dt) \
    case dt:                  \
        return std::to_string(*(to_cpp_type_t<dt> *)data_.data());

                DT_TO_STRING_CASE(dt_uint32);
                DT_TO_STRING_CASE(dt_float32);
                DT_TO_STRING_CASE(dt_bfloat16);
                DT_TO_STRING_CASE(dt_int32);
#undef DT_TO_STRING_CASE
            default:
                throw "un supported dtype to_string: " + std::string(nncase::datatype_names(dtype));
            }
        }
        else
        {
            return "[...]";
        }
    }

protected:
    bool properties_equal(node &other) const override;

private:
    std::vector<std::byte> data_;
    datatype_t datatype_;
};
}
