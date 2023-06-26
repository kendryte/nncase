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
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/type_serializer.h>
#include <utility>

using namespace nncase;
using namespace nncase::runtime;

namespace {
template <class TReader>
result<type> deserialize_type_impl(TReader &sr) noexcept {
    switch (sr.template read_unaligned<type_signature_token_t>()) {
    case type_sig_invalid:
        return ok<type>(invalid_type::value);
    case type_sig_any:
        return ok<type>(any_type::value);
    case type_sig_tensor: {
        checked_try_var(elem_type, deserialize_datatype(sr));
        shape_t shape(unranked_shape);
        auto is_scalar = sr.template read_unaligned<uint8_t>() == 0;
        if (is_scalar) {
            shape = scalar_shape;
        } else {
            uint8_t dim_token;
            while ((dim_token = sr.template read_unaligned<uint8_t>()) !=
                   type_sig_end) {
                if (dim_token == dim_fixed) {
                    shape.push_back(sr.template read_unaligned<uint32_t>());
                } else if (dim_token == dim_unknown) {
                    shape.push_back(unknown_dim);
                } else {
                    dbg("Invalid dim token: ", dim_token);
                    return err(std::errc::invalid_argument);
                }
            }
        }

        return ok<type>(tensor_type(std::in_place, elem_type, shape));
    }
    case type_sig_tuple: {
        itlib::small_vector<type> fields;
        type_signature_token_t type_token;
        while ((type_token =
                    sr.template peek_unaligned<type_signature_token_t>()) !=
               type_sig_end) {
            checked_try_var(field_type, deserialize_type(sr));
            fields.emplace_back(std::move(field_type));
        }

        sr.skip(sizeof(type_token));
        return ok<type>(tuple_type(std::in_place, std::move(fields)));
    }
    case type_sig_callable:
        return err(std::errc::not_supported);
    default:
        return err(std::errc::invalid_argument);
    }
}

template <class TReader>
result<datatype_t> deserialize_datatype_impl(TReader &sr) noexcept {
    auto typecode = sr.template read_unaligned<typecode_t>();
    switch (typecode) {
    case dt_pointer: {
        checked_try_var(elem_type, deserialize_datatype(sr));
        return ok<datatype_t>(pointer_type_t(std::in_place, elem_type));
    }
    case dt_valuetype: {
        auto uuid = sr.template read_unaligned<uuid_t>();
        auto size_bytes = sr.template read_unaligned<uint32_t>();
        return ok<datatype_t>(value_type_t(std::in_place, uuid, size_bytes));
    }
        // prim types
    default: {
        if (typecode >= dt_boolean && typecode <= dt_bfloat16) {
            return datatype_t::from_typecode(typecode);
        } else {
            return err(std::errc::invalid_argument);
        }
    }
    }
}
} // namespace

result<type> runtime::deserialize_type(span_reader &sr) noexcept {
    return deserialize_type_impl(sr);
}

result<datatype_t> runtime::deserialize_datatype(span_reader &sr) noexcept {
    return deserialize_datatype_impl(sr);
}

result<type> runtime::deserialize_type(stream_reader &sr) noexcept {
    return deserialize_type_impl(sr);
}

result<datatype_t> runtime::deserialize_datatype(stream_reader &sr) noexcept {
    return deserialize_datatype_impl(sr);
}
