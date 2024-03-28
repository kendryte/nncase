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
#include "runtime_function.h"
#include <chrono>
#include <ncnn/datareader.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

#ifdef ENABLE_OPENMP
#include <omp.h>
#define OMP_MAX_THREAD omp_get_max_threads()
#else
#define OMP_MAX_THREAD 1
#endif // ENABLE_OPENMP

size_t rdata_offset = 0;

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::ncnn;

namespace {
class DataReaderFromEmpty : public ::ncnn::DataReader {
  public:
    virtual int scan(const char *format, void *p) const { return 0; }
    virtual size_t read(void *buf, size_t size) const {
        memset(buf, 0, size);
        return size;
    }
};
} // namespace

ncnn_runtime_function::ncnn_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module) {}

ncnn_runtime_module &ncnn_runtime_function::module() const noexcept {
    return static_cast<ncnn_runtime_module &>(runtime_function::module());
}

result<void> ncnn_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    try_(context.read_section(".inputs",
                              [this](auto sr, size_t) -> result<void> {
                                  input_names_ = sr.read_string_array();
                                  return ok();
                              }));
    try_(context.read_section(".outputs",
                              [this](auto sr, size_t) -> result<void> {
                                  output_names_ = sr.read_string_array();
                                  return ok();
                              }));

    NNCASE_UNUSED stream_reader *sr = nullptr;
    section_header h;
    try_set(sr, context.seek_section(".rdata", h));
    auto param_mem = reinterpret_cast<const uint8_t *>(
        module().text().data() + context.header().entrypoint);
    if (context.header().entrypoint == 0)
        rdata_offset = 0;
    auto bin_mem = reinterpret_cast<const uint8_t *>(module().rdata().data() +
                                                     rdata_offset);

    // auto bin_mem = reinterpret_cast<const uint8_t *>(rdata_.data());
    ::ncnn::DataReaderFromMemory paramdr(param_mem);
    ::ncnn::DataReaderFromMemory bindr(bin_mem);

    CHECK_WITH_ERR(!net_.load_param(paramdr), std::errc::invalid_argument);
    CHECK_WITH_ERR(!net_.load_model(bindr), std::errc::invalid_argument);

    net_.opt.num_threads = OMP_MAX_THREAD;
    rdata_offset += h.memory_size;

    return ok();
}

result<value_t> ncnn_runtime_function::invoke_core(
    gsl::span<value_t> parameters,
    [[maybe_unused]] value_t return_value) noexcept {

    auto ex = net_.create_extractor();

    // 1. Set input
    for (size_t i = 0; i < parameters.size(); i++) {
        try_var(t, parameters[i].as<tensor>());
        ::ncnn::Mat mat;
        mat.elempack = 1;
        mat.elemsize = runtime::get_bytes(t->dtype());
        auto shape = t->shape();
        switch (shape.size()) {
        case 1:
            mat.dims = 1;
            mat.w = shape[0];
            mat.h = 1;
            mat.d = 1;
            mat.c = 1;
            mat.cstep = mat.w;
            break;
        case 2:
            mat.dims = 2;
            mat.w = shape[1];
            mat.h = shape[0];
            mat.d = 1;
            mat.c = 1;
            mat.cstep = (size_t)mat.w * mat.h;
            break;
        case 3:
            mat.dims = 3;
            mat.w = shape[2];
            mat.h = shape[1];
            mat.d = 1;
            mat.c = shape[0];
            mat.cstep = (size_t)mat.w * mat.h;
            break;
        case 4:
            mat.dims = 4;
            mat.w = shape[3];
            mat.h = shape[2];
            mat.d = shape[1];
            mat.c = shape[0];
            mat.cstep = (size_t)mat.w * mat.h * mat.d;
            break;
        default:
            return err(std::errc::invalid_argument);
        }

        try_var(hb, t->buffer().as_host());
        try_var(map, hb.map(map_read));
        mat.data = map.buffer().data();

        // Must clone to full fill ncnn implicit contracts:
        // 1. Input is not user mananged data
        // 2. Data must be aligned
        auto cloned_input = mat.clone();
        CHECK_WITH_ERR(!ex.input(input_names_[i].c_str(), cloned_input),
                       std::errc::invalid_argument);
    }

    // 2. Extract outputs
    std::vector<value_t> outputs;
    for (size_t i = 0; i < output_names_.size(); i++) {
        ::ncnn::Mat mat;
        CHECK_WITH_ERR(!ex.extract(output_names_[i].c_str(), mat),
                       std::errc::invalid_argument);
        auto mat_size = (size_t)mat.c * mat.w * mat.h * mat.d * mat.elemsize;
        gsl::span<gsl::byte> data{reinterpret_cast<gsl::byte *>(mat.data),
                                  mat_size};

        dims_t shape;
        switch (mat.dims) {
        case 1:
            shape = {(size_t)mat.w};
            break;
        case 2:
            shape = {(size_t)mat.h, (size_t)mat.w};
            break;
        case 3:
            shape = {(size_t)mat.c, (size_t)mat.h, (size_t)mat.w};
            break;
        case 4:
            shape = {(size_t)mat.c, (size_t)mat.d, (size_t)mat.h,
                     (size_t)mat.w};
            break;
        default:
            return err(std::errc::invalid_argument);
        }

        datatype_t dt;
        switch (mat.elemsize) {
        case 1:
            dt = datatype_t::uint8;
            break;
        case 2:
            dt = datatype_t::float16;
            break;
        case 4:
            dt = datatype_t::float32;
            break;
        default:
            return err(std::errc::invalid_argument);
        }

        buffer_allocate_options options{};
        options.flags = HOST_BUFFER_ALLOCATE_SHARED;
        try_var(buf, buffer_allocator::host().allocate(mat_size, options));
        try_var(hb, buf.as<host_buffer_t>());

        {
            try_var(map, hb->map(map_write));
            auto csize = (size_t)mat.w * mat.h * mat.d * mat.elemsize;
            for (size_t i = 0; i < (size_t)mat.c; i++) {
                void *dest =
                    (unsigned char *)map.buffer().data() + (size_t)i * csize;
                const void *src = (unsigned char *)mat.channel(i);
                memcpy(dest, src, csize);
            }
        }

        tensor t(std::in_place, dt, shape, get_default_strides(shape), buf);
        outputs.emplace_back(t);
    }
    auto ret_val = output_names_.size() == 1
                       ? outputs[0]
                       : tuple(std::in_place, std::move(outputs));

    if (!return_value.empty()) {
        try_(ret_val->copy_to(return_value));
        return ok(return_value);
    }

    return ok(ret_val);
}
