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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/result.h>

using namespace nncase;

prim_type_t datatype_t::boolean(std::in_place, dt_boolean);
prim_type_t datatype_t::uint8(std::in_place, dt_uint8);
prim_type_t datatype_t::uint16(std::in_place, dt_uint16);
prim_type_t datatype_t::uint32(std::in_place, dt_uint32);
prim_type_t datatype_t::uint64(std::in_place, dt_uint64);
prim_type_t datatype_t::int8(std::in_place, dt_int8);
prim_type_t datatype_t::int16(std::in_place, dt_int16);
prim_type_t datatype_t::int32(std::in_place, dt_int32);
prim_type_t datatype_t::int64(std::in_place, dt_int64);
prim_type_t datatype_t::float16(std::in_place, dt_float16);
prim_type_t datatype_t::float32(std::in_place, dt_float32);
prim_type_t datatype_t::float64(std::in_place, dt_float64);
prim_type_t datatype_t::bfloat16(std::in_place, dt_bfloat16);
prim_type_t datatype_t::float8e4m3(std::in_place, dt_float8e4m3);
prim_type_t datatype_t::float8e5m2(std::in_place, dt_float8e5m2);

result<prim_type_t> datatype_t::from_typecode(typecode_t typecode) {
    switch (typecode) {
    case dt_boolean:
        return ok(datatype_t::boolean);
    case dt_uint8:
        return ok(datatype_t::uint8);
    case dt_uint16:
        return ok(datatype_t::uint16);
    case dt_uint32:
        return ok(datatype_t::uint32);
    case dt_uint64:
        return ok(datatype_t::uint64);
    case dt_int8:
        return ok(datatype_t::int8);
    case dt_int16:
        return ok(datatype_t::int16);
    case dt_int32:
        return ok(datatype_t::int32);
    case dt_int64:
        return ok(datatype_t::int64);
    case dt_float16:
        return ok(datatype_t::float16);
    case dt_float32:
        return ok(datatype_t::float32);
    case dt_float64:
        return ok(datatype_t::float64);
    case dt_bfloat16:
        return ok(datatype_t::bfloat16);
    case dt_float8e4m3:
        return ok(datatype_t::float8e4m3);
    case dt_float8e5m2:
        return ok(datatype_t::float8e5m2);
    default:
        return err(std::errc::invalid_argument);
    }
}

datatype_t::datatype_t(typecode_t typecode)
    : datatype_t(from_typecode(typecode).unwrap()) {}
