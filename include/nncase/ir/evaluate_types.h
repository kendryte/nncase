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
#include <nncase/runtime/datatypes.h>

namespace nncase::ir
{
class NNCASE_API evaluate_tensor
{
public:
    evaluate_tensor(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> buffer);

    datatype_t datatype() const noexcept { return datatype_; }
    const runtime_shape_t &shape() const noexcept { return shape_; }
    const runtime_shape_t &strides() const noexcept { return strides_; }
    gsl::span<gsl::byte> buffer() const noexcept { return buffer_; }

private:
    datatype_t datatype_;
    runtime_shape_t shape_;
    runtime_shape_t strides_;
    gsl::span<gsl::byte> buffer_;
};
}
