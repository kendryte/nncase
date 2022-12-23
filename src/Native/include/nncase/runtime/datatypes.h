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
#include "../object.h"
#include "simple_types.h"

namespace nncase {
class NNCASE_API datatype_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_datatype)
  public:
    /** @brief Get size in bytes. */
    virtual size_t size_bytes() const noexcept = 0;

    /** @brief Get type code. */
    virtual typecode_t typecode() const noexcept = 0;
};

class prim_type_node;
using prim_type_t = object_t<prim_type_node>;

class NNCASE_API datatype_t : public object_t<datatype_node> {
  public:
    using object_t::object_t;

    static prim_type_t boolean;
    static prim_type_t uint8;
    static prim_type_t uint16;
    static prim_type_t uint32;
    static prim_type_t uint64;
    static prim_type_t int8;
    static prim_type_t int16;
    static prim_type_t int32;
    static prim_type_t int64;
    static prim_type_t float16;
    static prim_type_t float32;
    static prim_type_t float64;
    static prim_type_t bfloat16;

    datatype_t(typecode_t typecode);

    static result<prim_type_t> from_typecode(typecode_t typecode);

    template <class T> static datatype_t from_type();
};

class NNCASE_API prim_type_node : public datatype_node {
    DEFINE_OBJECT_KIND(datatype_node, object_prim_type)
  public:
    explicit prim_type_node(typecode_t typecode) noexcept
        : typecode_(typecode) {}

    size_t size_bytes() const noexcept override {
        return typecode_bytes(typecode_);
    }

    typecode_t typecode() const noexcept override { return typecode_; }

  private:
    typecode_t typecode_;
};

class NNCASE_API pointer_type_node : public datatype_node {
    DEFINE_OBJECT_KIND(datatype_node, object_pointer_type)
  public:
    explicit pointer_type_node(datatype_t elemtype) noexcept
        : elemtype_(elemtype) {}

    size_t size_bytes() const noexcept override {
        return typecode_bytes(dt_pointer);
    }

    typecode_t typecode() const noexcept override { return dt_pointer; }
    const datatype_t &elemtype() const noexcept { return elemtype_; }

  private:
    datatype_t elemtype_;
};

using pointer_type_t = object_t<pointer_type_node>;

class NNCASE_API value_type_node : public datatype_node {
    DEFINE_OBJECT_KIND(datatype_node, object_value_type)
  public:
    value_type_node(uuid_t uuid, size_t size_bytes) noexcept
        : uuid_(uuid), size_bytes_(size_bytes) {}

    size_t size_bytes() const noexcept override { return size_bytes_; }
    typecode_t typecode() const noexcept override { return dt_valuetype; }
    const uuid_t &uuid() const noexcept { return uuid_; }

  private:
    uuid_t uuid_;
    size_t size_bytes_;
};

using value_type_t = object_t<value_type_node>;

namespace detail {
template <class T> struct datatype_of {};

#define DEFINE_DATATYPE_OF(type, name)                                         \
    template <> struct datatype_of<type> {                                     \
        datatype_t operator()() const noexcept { return datatype_t::name; }    \
    };

DEFINE_DATATYPE_OF(bool, boolean)
DEFINE_DATATYPE_OF(uint8_t, uint8)
DEFINE_DATATYPE_OF(uint16_t, uint16)
DEFINE_DATATYPE_OF(uint32_t, uint32)
#ifdef __APPLE__
DEFINE_DATATYPE_OF(size_t, uint64)
#endif
DEFINE_DATATYPE_OF(uint64_t, uint64)
DEFINE_DATATYPE_OF(int8_t, int8)
DEFINE_DATATYPE_OF(int16_t, int16)
DEFINE_DATATYPE_OF(int32_t, int32)
DEFINE_DATATYPE_OF(int64_t, int64)
DEFINE_DATATYPE_OF(half, float16)
DEFINE_DATATYPE_OF(float, float32)
DEFINE_DATATYPE_OF(double, float64)
DEFINE_DATATYPE_OF(bfloat16, bfloat16)

#undef DEFINE_DATATYPE_OF
} // namespace detail

template <class T> datatype_t datatype_t::from_type() {
    return detail::datatype_of<T>()();
}

inline result<typecode_t> to_typecode(const datatype_t &dtype) {
    try_var(prim_type, dtype.as<prim_type_t>());
    return ok(prim_type->typecode());
}
} // namespace nncase
