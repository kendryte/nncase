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
#include <atomic>
#include <nncase/compiler.h>
#include <nncase/compiler_defs.h>
#include <pybind11/pybind11.h>

namespace pybind11::detail {
extern std::atomic_bool g_python_shutdown;

inline bool is_py_shutdown() {
    return !Py_IsInitialized() ||
           g_python_shutdown.load(std::memory_order_acquire);
}

template <> struct type_caster<std::span<const std::byte>> {
  public:
    PYBIND11_TYPE_CASTER(std::span<const std::byte>, _("bytes"));

    bool load(handle src, bool) {
        if (!py::isinstance<py::bytes>(src))
            return false;

        uint8_t *buffer;
        py::ssize_t length;
        if (PyBytes_AsStringAndSize(
                src.ptr(), reinterpret_cast<char **>(&buffer), &length))
            return false;
        value = {(const std::byte *)buffer, (size_t)length};
        loader_life_support::add_patient(src);
        return true;
    }
};

#define NNCASE_CSTREAM_IMPL_COMMON                                             \
    py::gil_scoped_acquire gil;                                                \
    py::handle pyhandle(reinterpret_cast<PyObject *>(handle))

template <> struct type_caster<nncase::clr::cstream> {
  public:
    PYBIND11_TYPE_CASTER(nncase::clr::cstream, _("CStream"));
    inline static nncase_stream_mt_t _mt = {
        .add_ref =
            [](nncase_stream_handle_t handle) {
                NNCASE_CSTREAM_IMPL_COMMON;
                pyhandle.inc_ref();
            },
        .release =
            [](nncase_stream_handle_t handle) {
                if (is_py_shutdown())
                    return;
                NNCASE_CSTREAM_IMPL_COMMON;
                pyhandle.dec_ref();
            },
        .can_read =
            [](nncase_stream_handle_t handle) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("readable")().cast<bool>();
            },
        .can_seek =
            [](nncase_stream_handle_t handle) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("seekable")().cast<bool>();
            },
        .can_write =
            [](nncase_stream_handle_t handle) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("writable")().cast<bool>();
            },
        .flush =
            [](nncase_stream_handle_t handle) {
                if (is_py_shutdown())
                    return;
                NNCASE_CSTREAM_IMPL_COMMON;
                pyhandle.attr("flush")();
            },
        .get_length =
            [](nncase_stream_handle_t handle) {
                NNCASE_CSTREAM_IMPL_COMMON;
                auto cur = pyhandle.attr("tell")();
                auto seek = pyhandle.attr("seek");
                auto len = seek(0, 2).cast<int64_t>(); // seek end
                seek(cur, 0);
                return len;
            },
        .set_length =
            [](nncase_stream_handle_t handle, uint64_t value) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("truncate")(value).cast<int64_t>();
            },
        .get_position =
            [](nncase_stream_handle_t handle) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("tell")().cast<int64_t>();
            },
        .read =
            [](nncase_stream_handle_t handle, uint8_t *buffer, size_t length) {
                NNCASE_CSTREAM_IMPL_COMMON;
                auto mem =
                    py::memoryview::from_memory(buffer, py::ssize_t(length));
                return pyhandle.attr("readinto")(mem).cast<size_t>();
            },
        .seek =
            [](nncase_stream_handle_t handle, int origin, int64_t offset) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("seek")(origin, offset).cast<int64_t>();
            },
        .write =
            [](nncase_stream_handle_t handle, const uint8_t *buffer,
               size_t length) {
                NNCASE_CSTREAM_IMPL_COMMON;
                auto mem =
                    py::memoryview::from_memory(buffer, py::ssize_t(length));
                pyhandle.attr("write")(mem);
            }};

    type_caster() : value(nullptr) {}

    bool load(handle src, bool) {
        if (getattr(src, "write", py::none()).is_none())
            return false;

        obj = py::reinterpret_borrow<object>(src);
        new (&value) nncase::clr::cstream(&_mt, src.ptr());
        return true;
    }

  protected:
    py::object obj;
};

#undef NNCASE_CSTREAM_IMPL_COMMON
} // namespace pybind11::detail
