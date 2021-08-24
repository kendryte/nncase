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
#include "nncase/ir/ir_types.h"
#include "nncase/runtime/datatypes.h"

namespace nncase::ir
{
/** @brief Expression type  **/
class NNCASE_API type
{
public:
    /** @brief Initialize a scalar type **/
    type(datatype_t elem_type) noexcept
        : elem_type_(elem_type)
    {
    }

    /** @brief Initialize a scalar/tensor type **/
    type(datatype_t elem_type, shape_t shape) noexcept
        : elem_type_(elem_type), shape_(std::move(shape))
    {
    }

    /** @brief Is this a scalar type **/
    bool is_scalar() const noexcept { return shape_.empty(); }

    /** @brief Is this a tensor type **/
    bool is_tensor() const noexcept { return !shape_.empty(); }

    /** @brief Get element datatype **/
    datatype_t elem_type() const noexcept { return elem_type_; }
    /** @brief Set element datatype **/
    void elem_type(datatype_t value) noexcept { elem_type_ = value; }

    /** @brief Get shape **/
    const shape_t &shape() const noexcept { return shape_; }
    /** @brief Get mutable shape **/
    shape_t &shape() noexcept { return shape_; }
    /** @brief Set shape **/
    void shape(shape_t value) noexcept { shape_ = std::move(value); }

private:
    datatype_t elem_type_;
    shape_t shape_;
};
}
