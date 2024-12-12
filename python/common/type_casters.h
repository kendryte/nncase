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
#include <nncase/runtime/stream.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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
            [](nncase_stream_handle_t handle, int64_t offset, int origin) {
                NNCASE_CSTREAM_IMPL_COMMON;
                return pyhandle.attr("seek")(offset, origin).cast<int64_t>();
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

class pystream : public nncase::runtime::stream {
  public:
    pystream(py::object stream)
        : stream_(std::move(stream)),
          py_readinto_(getattr(stream_, "readinto", py::none())),
          py_write_(getattr(stream_, "write", py::none())),
          py_seek_(getattr(stream_, "seek", py::none())),
          py_tell_(getattr(stream_, "tell", py::none())) {
        /* Some Python file objects (e.g. sys.stdout and sys.stdin)
         have non-functional seek and tell. If so, assign None to
         py_tell and py_seek.
       */
        if (!py_tell_.is_none()) {
            try {
                py_tell_();
            } catch (py::error_already_set &err) {
                py_tell_ = py::none();
                py_seek_ = py::none();
                err.restore();
                PyErr_Clear();
            }
        }
    }

    nncase::result<std::streampos> tell() const noexcept override {
        if (!py_tell_.is_none()) {
            try {
                return nncase::ok(py_tell_().cast<std::streamoff>());
            } catch (...) {
            }
        } else {
            return nncase::err(std::errc::not_supported);
        }
        return nncase::err(std::errc::io_error);
    }

    nncase::result<void> seek(std::streamoff offset,
                              std::ios::seekdir dir) noexcept override {
        if (!py_seek_.is_none()) {
            try {
                py_seek_((int64_t)offset, (int)dir);
                return nncase::ok();
            } catch (...) {
            }
        } else {
            return nncase::err(std::errc::not_supported);
        }
        return nncase::err(std::errc::io_error);
    }

    nncase::result<size_t> read(void *buffer, size_t bytes) noexcept override {
        if (!py_readinto_.is_none()) {
            try {
                auto mem =
                    py::memoryview::from_memory(buffer, py::ssize_t(bytes));
                return nncase::ok(py_readinto_(mem).cast<size_t>());
            } catch (...) {
            }
        } else {
            return nncase::err(std::errc::not_supported);
        }
        return nncase::err(std::errc::io_error);
    }

    nncase::result<void> write(const void *buffer,
                               size_t bytes) noexcept override {
        if (!py_write_.is_none()) {
            try {
                auto mem =
                    py::memoryview::from_memory(buffer, py::ssize_t(bytes));
                py_write_(mem);
                return nncase::ok();
            } catch (...) {
            }
        } else {
            return nncase::err(std::errc::not_supported);
        }
        return nncase::err(std::errc::io_error);
    }

  private:
    py::object stream_;
    py::object py_readinto_;
    py::object py_write_;
    py::object py_seek_;
    py::object py_tell_;
};

template <> struct type_caster<nncase::runtime::stream> {
  public:
    bool load(handle src, bool) {
        if (getattr(src, "read", py::none()).is_none() &&
            getattr(src, "write", py::none()).is_none()) {
            return false;
        }

        obj = py::reinterpret_borrow<object>(src);
        value = std::make_unique<pystream>(obj);

        return true;
    }

  protected:
    py::object obj;
    std::unique_ptr<nncase::runtime::stream> value;

  public:
    static constexpr auto name = _("pystream");
    static handle cast(NNCASE_UNUSED const nncase::runtime::stream *src,
                       NNCASE_UNUSED return_value_policy policy,
                       NNCASE_UNUSED handle parent) {
        return none().release();
    }
    operator nncase::runtime::stream *() { return value.get(); }
    operator nncase::runtime::stream &() { return *value; }
    template <typename _T>
    using cast_op_type = pybind11::detail::cast_op_type<_T>;
};
} // namespace pybind11::detail
