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
#include <nncase/compiler_defs.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/value.h>

extern "C" {
typedef void *clr_object_handle_t;
typedef void *nncase_stream_handle_t;

typedef enum {
    nncase_array_rtvalue = 0,
    nncase_array_var = 1
} nncase_array_element_kind_t;

typedef enum {
    nncase_qm_unsigned = 0,
    nncase_qm_signed_symmetric = 1,
    nncase_qm_signed_asymmetric = 2
} nncase_quant_mode_t;

typedef enum {
    nncase_calib_noclip = 0,
    nncase_calib_kld = 1
} nncase_calib_method_t;

typedef struct {
    void (*add_ref)(nncase_stream_handle_t handle);
    void (*release)(nncase_stream_handle_t handle);
    bool (*can_read)(nncase_stream_handle_t handle);
    bool (*can_seek)(nncase_stream_handle_t handle);
    bool (*can_write)(nncase_stream_handle_t handle);
    void (*flush)(nncase_stream_handle_t handle);
    int64_t (*get_length)(nncase_stream_handle_t handle);
    int64_t (*set_length)(nncase_stream_handle_t handle, uint64_t value);
    int64_t (*get_position)(nncase_stream_handle_t handle);
    size_t (*read)(nncase_stream_handle_t handle, uint8_t *buffer,
                   size_t length);
    int64_t (*seek)(nncase_stream_handle_t handle, int origin, int64_t offset);
    void (*write)(nncase_stream_handle_t handle, const uint8_t *buffer,
                  size_t length);
} nncase_stream_mt_t;

NNCASE_API int nncase_clr_initialize(const char *root_assembly_path);

NNCASE_API int nncase_clr_array_create(nncase_array_element_kind_t kind,
                                       const clr_object_handle_t *elements,
                                       size_t count,
                                       clr_object_handle_t *array);
NNCASE_API int nncase_clr_array_get_item(clr_object_handle_t array,
                                         size_t index,
                                         clr_object_handle_t *item);
NNCASE_API int nncase_clr_array_get_length(clr_object_handle_t array,
                                           size_t *length);

NNCASE_API int nncase_clr_calibration_dataset_provider_create(
    clr_object_handle_t dataset, size_t samplesCount,
    clr_object_handle_t fn_params, clr_object_handle_t *provider);

NNCASE_API int nncase_clr_handle_free(clr_object_handle_t handle);

NNCASE_API int
nncase_clr_compile_options_create(clr_object_handle_t *compile_options);
NNCASE_API int
nncase_clr_compile_options_set_input_file(clr_object_handle_t compile_options,
                                          const char *input_file,
                                          size_t input_file_length);
NNCASE_API int
nncase_clr_compile_options_set_input_format(clr_object_handle_t compile_options,
                                            const char *input_format,
                                            size_t input_format_length);
NNCASE_API int
nncase_clr_compile_options_set_target(clr_object_handle_t compile_options,
                                      const char *target, size_t target_length);
NNCASE_API int
nncase_clr_compile_options_set_dump_level(clr_object_handle_t compile_options,
                                          int32_t dump_level);
NNCASE_API int
nncase_clr_compile_options_set_dump_dir(clr_object_handle_t compile_options,
                                        const char *dump_dir,
                                        size_t dump_dir_length);
NNCASE_API int nncase_clr_compile_options_set_quantize_options(
    clr_object_handle_t compile_options, clr_object_handle_t quantize_options);
NNCASE_API int
nncase_clr_compile_options_set_quant_type(clr_object_handle_t compile_options,
                                          clr_object_handle_t quant_type);
NNCASE_API int
nncase_clr_compile_options_set_quant_mode(clr_object_handle_t compile_options,
                                          nncase_quant_mode_t quant_mode);

NNCASE_API int nncase_clr_compiler_create(clr_object_handle_t compile_options,
                                          clr_object_handle_t *compiler);
NNCASE_API int nncase_clr_compiler_import_module(clr_object_handle_t compiler,
                                                 clr_object_handle_t stream,
                                                 clr_object_handle_t *module);
NNCASE_API int nncase_clr_compiler_compile(clr_object_handle_t compiler);
NNCASE_API int nncase_clr_compiler_gencode(clr_object_handle_t compiler,
                                           clr_object_handle_t stream);

NNCASE_API int nncase_clr_datatype_from_typecode(nncase::typecode_t typecode,
                                                 clr_object_handle_t *datatype);

NNCASE_API int nncase_clr_expr_evaluate(clr_object_handle_t expr,
                                        clr_object_handle_t parameters,
                                        clr_object_handle_t inputs,
                                        clr_object_handle_t *result);

NNCASE_API int nncase_clr_function_get_body(clr_object_handle_t function,
                                            clr_object_handle_t *body);
NNCASE_API int
nncase_clr_function_get_parameters(clr_object_handle_t function,
                                   clr_object_handle_t *parameters);
NNCASE_API int nncase_clr_ir_module_get_entry(clr_object_handle_t module,
                                              clr_object_handle_t *entry);

NNCASE_API int nncase_clr_launch_debugger();

NNCASE_API int
nncase_clr_quantize_options_create(clr_object_handle_t *quantize_options);
NNCASE_API int nncase_clr_quantize_options_set_calibration_dataset(
    clr_object_handle_t quantize_options, clr_object_handle_t dataset);
NNCASE_API int nncase_clr_quantize_options_set_calibration_method(
    clr_object_handle_t quantize_options, nncase_calib_method_t method);

NNCASE_API int nncase_clr_rtvalue_from_handle(nncase::value_node *value,
                                              clr_object_handle_t *rtvalue);
NNCASE_API int nncase_clr_rtvalue_get_handle(clr_object_handle_t rtvalue,
                                             nncase::value_node **value);

NNCASE_API int nncase_clr_stream_create(const nncase_stream_mt_t *mt,
                                        void *handle,
                                        clr_object_handle_t *stream);

NNCASE_API bool nncase_clr_target_exists(const char *target_name,
                                         size_t target_name_length);
}

namespace nncase::clr {
class clr_object_ptr {
  public:
    constexpr clr_object_ptr(std::nullptr_t = nullptr) noexcept
        : handle_(nullptr) {}
    constexpr clr_object_ptr(clr_object_handle_t handle) noexcept
        : handle_(handle) {}
    constexpr clr_object_ptr(clr_object_ptr &&other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    ~clr_object_ptr() { release(); }

    clr_object_ptr(const clr_object_ptr &) = delete;
    clr_object_ptr &operator=(const clr_object_ptr &) = delete;

    clr_object_ptr &operator=(clr_object_ptr &&other) noexcept {
        release();
        handle_ = other.handle_;
        other.handle_ = nullptr;
        return *this;
    }

    clr_object_handle_t get() const noexcept { return handle_; }

    clr_object_handle_t detach() noexcept {
        auto handle = handle_;
        handle_ = nullptr;
        return handle;
    }

    clr_object_handle_t *release_and_addressof() noexcept {
        release();
        return &handle_;
    }

  private:
    void release() {
        if (auto handle = handle_) {
            handle_ = nullptr;
            nncase_clr_handle_free(handle);
        }
    }

  private:
    clr_object_handle_t handle_;
};

#define CHECK_CLR(x)                                                           \
    if (x) {                                                                   \
        throw std::runtime_error(#x);                                          \
    }

class clr_object_base {
  public:
    constexpr clr_object_base(std::nullptr_t = nullptr) noexcept
        : obj_(nullptr) {}

    clr_object_base(clr_object_base &&) = default;
    clr_object_base &operator=(clr_object_base &&) = default;

    clr_object_handle_t get() const noexcept { return obj_.get(); }
    clr_object_handle_t *release_and_addressof() noexcept {
        return obj_.release_and_addressof();
    }

    template <class T, std::enable_if_t<std::is_base_of_v<clr_object_base, T>>>
    T &cast() noexcept {
        return static_cast<T &>(*this);
    }

  protected:
    clr_object_ptr obj_;
};

class quantize_options : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    quantize_options() {
        CHECK_CLR(
            nncase_clr_quantize_options_create(obj_.release_and_addressof()));
    }
};

class cstream : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    cstream(const nncase_stream_mt_t *mt, void *handle) {
        CHECK_CLR(
            nncase_clr_stream_create(mt, handle, obj_.release_and_addressof()));
    }
};

class compile_options : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    compile_options() {
        CHECK_CLR(
            nncase_clr_compile_options_create(obj_.release_and_addressof()));
    }

    std::string input_format() { return "cpu"; }
    void input_format(std::string_view value) {
        CHECK_CLR(nncase_clr_compile_options_set_input_format(
            obj_.get(), value.data(), value.length()));
    }

    std::string target() { return "cpu"; }
    void target(std::string_view value) {
        CHECK_CLR(nncase_clr_compile_options_set_target(
            obj_.get(), value.data(), value.length()));
    }

    int32_t dump_level() { return 0; }
    void dump_level(int32_t value) {
        CHECK_CLR(nncase_clr_compile_options_set_dump_level(obj_.get(), value));
    }

    std::string dump_dir() { return "cpu"; }
    void dump_dir(std::string_view value) {
        CHECK_CLR(nncase_clr_compile_options_set_dump_dir(
            obj_.get(), value.data(), value.length()));
    }
};

class array : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    array(nncase_array_element_kind_t kind, const clr_object_handle_t *elements,
          size_t length) {
        CHECK_CLR(nncase_clr_array_create(kind, elements, length,
                                          obj_.release_and_addressof()));
    }

    template <class T = clr_object_base> T at(size_t index) {
        T value(nullptr);
        CHECK_CLR(nncase_clr_array_get_item(obj_.get(), index,
                                            value.release_and_addressof()));
        return value;
    }

    size_t length() {
        size_t length;
        CHECK_CLR(nncase_clr_array_get_length(obj_.get(), &length));
        return length;
    }

    template <class T = clr_object_base> std::vector<T> to_vector() {
        std::vector<T> vector(length());
        for (size_t i = 0; i < vector.size(); i++) {
            vector[i] = at<T>(i);
        }
        return vector;
    }
};

class rtvalue : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    rtvalue(nncase::value_t value) {
        CHECK_CLR(nncase_clr_rtvalue_from_handle(value.detach(),
                                                 obj_.release_and_addressof()));
    }

    value_t to_value() const {
        value_t value(nullptr);
        CHECK_CLR(nncase_clr_rtvalue_get_handle(obj_.get(),
                                                value.release_and_addressof()));
        return value;
    }
};

class expr : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    rtvalue evaluate(const array &params, const array &inputs) {
        rtvalue value(nullptr);
        CHECK_CLR(nncase_clr_expr_evaluate(get(), params.get(), inputs.get(),
                                           value.release_and_addressof()));
        return value;
    }
};

class var : public expr {
  public:
    using expr::expr;
};

class base_function : public expr {
  public:
    using expr::expr;
};

class function : public base_function {
  public:
    using base_function::base_function;

    expr body() {
        expr body(nullptr);
        CHECK_CLR(
            nncase_clr_function_get_body(get(), body.release_and_addressof()));
        return body;
    }

    array parameters() {
        array params(nullptr);
        CHECK_CLR(nncase_clr_function_get_parameters(
            get(), params.release_and_addressof()));
        return params;
    }
};

class ir_module : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    function entry() {
        function entry(nullptr);
        CHECK_CLR(nncase_clr_ir_module_get_entry(
            get(), entry.release_and_addressof()));
        return entry;
    }
};

class compiler : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    compiler(const compile_options &options) {
        CHECK_CLR(nncase_clr_compiler_create(options.get(),
                                             obj_.release_and_addressof()));
    }

    ir_module import_module(cstream &stream) {
        ir_module module(nullptr);
        CHECK_CLR(nncase_clr_compiler_import_module(
            get(), stream.get(), module.release_and_addressof()));
        return module;
    }

    void compile() { CHECK_CLR(nncase_clr_compiler_compile(obj_.get())); }
    void gencode(cstream &stream) {
        CHECK_CLR(nncase_clr_compiler_gencode(obj_.get(), stream.get()));
    }
};
} // namespace nncase::clr
