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
#include "shape.h"
#include <nncase/runtime/datatypes.h>

namespace nncase {
/** @brief Type node */
class NNCASE_API type_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_type);

  protected:
    type_node() = default;
};

/** @brief Type */
using type = object_t<type_node>;

class any_type_node;

/** @brief Any type */
class any_type : public object_t<any_type_node> {
  public:
    using object_t::object_t;

    static any_type value;
};

/** @brief Any type node */
class NNCASE_API any_type_node : public type_node {
    DEFINE_OBJECT_KIND(object_node, object_any_type);

  public:
};

class invalid_type_node;

/** @brief Invalid type */
class invalid_type : public object_t<invalid_type_node> {
  public:
    using object_t::object_t;

    static invalid_type value;
};

/** @brief Invalid type node */
class NNCASE_API invalid_type_node : public type_node {
    DEFINE_OBJECT_KIND(object_node, object_invalid_type);

  public:
    invalid_type_node() noexcept {}
    invalid_type_node(std::string reason) noexcept
        : reason_(std::move(reason)) {}

    /** @brief Get reason */
    const std::string &reason() const noexcept { return reason_; }
    /** @brief Get mutable reason */
    std::string &reason() noexcept { return reason_; }
    /** @brief Set reason */
    void reason(std::string value) noexcept { reason_ = std::move(value); }

  private:
    std::string reason_;
};

/** @brief Tensor type node */
class NNCASE_API tensor_type_node : public type_node {
    DEFINE_OBJECT_KIND(object_node, object_tensor_type);

  public:
    tensor_type_node(datatype_t dtype, shape_t shape) noexcept
        : dtype_(std::move(dtype)), shape_(std::move(shape)) {}

    /** @brief Is this a scalar type */
    bool is_scalar() const noexcept { return shape_.is_scalar(); }
    /** @brief Is this a tensor type */
    bool is_tensor() const noexcept { return !shape_.is_scalar(); }

    /** @brief Get element datatype */
    datatype_t dtype() const noexcept { return dtype_; }
    /** @brief Set element datatype */
    void dtype(datatype_t value) noexcept { dtype_ = value; }

    /** @brief Get shape */
    const shape_t &shape() const noexcept { return shape_; }
    /** @brief Get mutable shape */
    shape_t &shape() noexcept { return shape_; }
    /** @brief Set shape */
    void shape(shape_t value) noexcept { shape_ = std::move(value); }

  private:
    datatype_t dtype_;
    shape_t shape_;
};

/** @brief Tensor type */
using tensor_type = object_t<tensor_type_node>;

/** @brief Tuple type node */
class NNCASE_API tuple_type_node : public type_node {
    DEFINE_OBJECT_KIND(object_node, object_tuple_type);

  public:
    tuple_type_node(itlib::small_vector<type> fields) noexcept
        : fields_(std::move(fields)) {}

    /** @brief Get fields */
    gsl::span<const type> fields() const noexcept { return fields_; }
    /** @brief Get mutable fields */
    itlib::small_vector<type> &shape() noexcept { return fields_; }
    /** @brief Set fields */
    void shape(itlib::small_vector<type> value) noexcept {
        fields_ = std::move(value);
    }

  private:
    itlib::small_vector<type> fields_;
};

/** @brief Tuple type */
using tuple_type = object_t<tuple_type_node>;

/** @brief Callable type node */
class NNCASE_API callable_type_node : public type_node {
    DEFINE_OBJECT_KIND(object_node, object_callable_type);

  public:
    callable_type_node(itlib::small_vector<type> parameters,
                       type return_type) noexcept
        : parameters_(std::move(parameters)), return_type_(return_type) {}

    /** @brief Get parameters */
    gsl::span<const type> parameters() const noexcept { return parameters_; }
    /** @brief Get parameters */
    itlib::small_vector<type> &parameters() noexcept { return parameters_; }
    /** @brief Set parameters */
    void parameters(itlib::small_vector<type> value) noexcept {
        parameters_ = std::move(value);
    }

    /** @brief Get return type */
    const type &return_type() const noexcept { return return_type_; }
    /** @brief Get mutable return type */
    type &return_type() noexcept { return return_type_; }
    /** @brief Set return type */
    void return_type(type value) noexcept { return_type_ = std::move(value); }

  private:
    itlib::small_vector<type> parameters_;
    type return_type_;
};

/** @brief Callable type */
using callable_type = object_t<callable_type_node>;
} // namespace nncase
