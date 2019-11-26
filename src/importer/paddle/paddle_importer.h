/* Copyright 2019 Canaan Inc.
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
#include "framework.pb.h"
#include <boost/filesystem.hpp>
#include <ir/connectors.h>
#include <ir/graph.h>
#include <ir/op_utils.h>
#include <ir/ops/transpose.h>
#include <unordered_map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace importer
{
    class paddle_importer
    {
    public:
        paddle_importer(xtl::span<const uint8_t> model, const boost::filesystem::path &params_dir, ir::graph &graph);

        void import();

    private:
        void convert_op(const paddle::framework::proto::OpDesc &op);

#define DEFINE_OPCODE(opcode) void convert_op_##opcode(const paddle::framework::proto::OpDesc &op);
#include "opcode.def"
#undef DEFINE_OPCODE

        const paddle::framework::proto::VarDesc &find_var(std::string_view name) const;
        const paddle::framework::proto::VarDesc &find_var(const google::protobuf::RepeatedPtrField<paddle::framework::proto::OpDesc_Var> &container, std::string_view name) const;

        constexpr datatype_t to_data_type(paddle::framework::proto::VarType_Type type)
        {
            using namespace paddle::framework::proto;

            switch (type)
            {
            case VarType_Type_FP32:
                return dt_float32;
            default:
                throw std::runtime_error("Invalid tesnor type");
            }
        }

        template <class T>
        constexpr paddle::framework::proto::VarType_Type to_tensor_type()
        {
            using namespace paddle::framework::proto;

            if constexpr (std::is_same_v<T, float>)
                return VarType_Type_FP32;
            else
                assert(!"Invalid element type");
        }

        template <size_t N>
        std::array<size_t, N> to_tensor_shape(const ir::shape_t &shape)
        {
            if (shape.size() != N)
                throw std::runtime_error("Invalid tesnor ranks");
            std::array<size_t, N> arr;
            std::copy_n(shape.begin(), N, arr.begin());
            return arr;
        }

        template <class T>
        xt::xarray<T> load_array(const paddle::framework::proto::VarDesc &var)
        {
            xt::xarray<T> arr(get_var_shape(var));
            if (get_lod_tensor_type(var) != to_tensor_type<T>())
                throw std::runtime_error("Unexpected tensor type");
            load_tensor(var.name(), reinterpret_cast<uint8_t *>(arr.begin()), reinterpret_cast<uint8_t *>(arr.end()));
            return std::move(arr);
        }

        template <class T, size_t N>
        xt::xtensor<T, N> load_tensor(const paddle::framework::proto::VarDesc &var)
        {
            xt::xtensor<T, N> tensor(to_tensor_shape<N>(get_var_shape(var)));
            if (get_lod_tensor_type(var) != to_tensor_type<T>())
                throw std::runtime_error("Unexpected tensor type");
            load_tensor(var.name(), reinterpret_cast<uint8_t *>(tensor.begin()), reinterpret_cast<uint8_t *>(tensor.end()));
            return std::move(tensor);
        }

        void load_tensor(std::string_view name, uint8_t *begin, uint8_t *end);

    private:
        static const paddle::framework::proto::OpDesc_Var &find_param(const google::protobuf::RepeatedPtrField<paddle::framework::proto::OpDesc_Var> &container, std::string_view name);
        static ir::shape_t get_var_shape(const paddle::framework::proto::VarDesc &var);
        static const paddle::framework::proto::OpDesc_Attr &find_attr(const google::protobuf::RepeatedPtrField<paddle::framework::proto::OpDesc_Attr> &container, std::string_view name);

        static paddle::framework::proto::VarType_Type get_lod_tensor_type(const paddle::framework::proto::VarDesc &var)
        {
            assert(var.type().type() == paddle::framework::proto::VarType_Type_LOD_TENSOR);
            return var.type().lod_tensor().tensor().data_type();
        }

    private:
        paddle::framework::proto::ProgramDesc model_;
        boost::filesystem::path params_dir_;
        ir::graph &graph_;
        const paddle::framework::proto::BlockDesc *subgraph_;
        std::unordered_map<ir::input_connector *, std::string_view> input_tensors_;
        std::unordered_map<std::string_view, ir::output_connector *> output_tensors_;
    };
}
}

#define DEFINE_PADDLE_LOWER(opcode) \
    void nncase::importer::paddle_importer::convert_op_##opcode(const paddle::framework::proto::OpDesc &op)
