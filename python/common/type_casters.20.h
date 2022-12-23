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
#include <nncase/runtime/compiler_defs.h>
#include <pybind11/pybind11.h>
#include <span>

namespace pybind11 {
namespace detail {
template <> struct type_caster<std::span<const uint8_t>> {
  public:
    PYBIND11_TYPE_CASTER(std::span<const uint8_t>, _("bytes"));

    bool load(handle src, bool) {
        if (!py::isinstance<py::bytes>(src))
            return false;

        uint8_t *buffer;
        py::ssize_t length;
        if (PyBytes_AsStringAndSize(
                src.ptr(), reinterpret_cast<char **>(&buffer), &length))
            return false;
        value = {buffer, (size_t)length};
        return true;
    }
};
} // namespace detail
} // namespace pybind11
