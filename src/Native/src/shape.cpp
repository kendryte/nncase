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
#include <nncase/shape.h>

using namespace nncase;

void shape_t::update_kind(shape_kind_t before_kind,
                          dim_kind_t new_dim_kind) noexcept {
    if (before_kind == shape_kind_fixed && new_dim_kind == dim_unknown) {
        kind_ = shape_kind_has_unknown_dim;
    } else if ((before_kind == shape_kind_has_unknown_dim &&
                new_dim_kind == dim_fixed) ||
               before_kind == shape_kind_unranked) {
        kind_ = kind_of(dims_);
    }
}

void shape_t::dim(size_t index, dim_t value) {
    auto before_kind = kind();
    dims_.at(index) = value;
    update_kind(before_kind, value.kind);
}

void shape_t::push_back(dim_t value) { emplace_back(value); }

const dim_t &shape_t::emplace_back(dim_t value) {
    auto before_kind = kind();
    auto &dim = dims_.emplace_back(value);
    update_kind(before_kind, value.kind);
    return dim;
}

const dim_t *shape_t::emplace(const dim_t *position, dim_t value) {
    auto before_kind = kind();
    auto dim = dims_.emplace(position, value);
    update_kind(before_kind, value.kind);
    return dim;
}

void shape_t::pop_back() {
    auto before_kind = kind();
    dims_.pop_back();
    if (before_kind == shape_kind_has_unknown_dim) {
        kind_ = kind_of(dims_);
    }
}

result<dims_t> shape_t::as_fixed() const noexcept {
    if (!is_fixed()) {
        return err(std::errc::invalid_argument);
    }

    dims_t dims(rank().value());
    for (size_t i = 0; i < dims.size(); i++) {
        dims[i] = dims_[i].fixed_value();
    }

    return ok(std::move(dims));
}
