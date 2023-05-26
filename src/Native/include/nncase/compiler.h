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
#include <cstring>
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
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
    nncase_mqm_no_quant = 0,
    nncase_mqm_use_ptq = 1,
    nncase_mqm_use_qat = 2
} nncase_model_quant_mode_t;

typedef enum {
    nncase_qm_unsigned = 0,
    nncase_qm_signed_symmetric = 1,
    nncase_qm_signed_asymmetric = 2
} nncase_quant_mode_t;

typedef enum {
    nncase_qt_uint8 = 0,
    nncase_qt_int8 = 1,
    nncase_qt_int16 = 2
} nncase_quant_type_t;

typedef enum {
    nncase_calib_noclip = 0,
    nncase_calib_kld = 1
} nncase_calib_method_t;

typedef enum {
    nncase_no_finetune_weights = 0,
    nncase_finetune_weights_squant = 1,
    nncase_finetune_weights_adaround = 2
} nncase_finetune_weights_method_t;

typedef enum {
    nncase_dump_flags_none = 0,
    nncase_dump_flags_import_ops = 1 << 1,
    nncase_dump_flags_pass_ir = 1 << 2,
    nncase_dump_flags_egraph_cost = 1 << 3,
    nncase_dump_flags_rewrite = 1 << 4,
    nncase_dump_flags_calibration = 1 << 5,
    nncase_dump_flags_evaluator = 1 << 6,
    nncase_dump_flags_compile = 1 << 7,
    nncase_dump_flags_tiling = 1 << 8,
    nncase_dump_flags_schedule = 1 << 9,
    nncase_dump_flags_codegen = 1 << 10
} nncase_dump_flags_t;

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

typedef struct {
    clr_object_handle_t (*array_create)(nncase_array_element_kind_t kind,
                                        const clr_object_handle_t *elements,
                                        size_t count);
    clr_object_handle_t (*array_get_item)(clr_object_handle_t array,
                                          size_t index);
    size_t (*array_get_length)(clr_object_handle_t array);
    clr_object_handle_t (*calibration_dataset_provider_create)(
        clr_object_handle_t dataset, size_t samplesCount,
        clr_object_handle_t fn_params);
    void (*handle_free)(clr_object_handle_t handle);
    clr_object_handle_t (*compile_options_create)();
    void (*compile_options_set_input_file)(clr_object_handle_t compile_options,
                                           const char *input_file,
                                           size_t input_file_length);
    void (*compile_options_set_input_format)(
        clr_object_handle_t compile_options, const char *input_format,
        size_t input_format_length);
    void (*compile_options_set_dump_dir)(clr_object_handle_t compile_options,
                                         const char *dump_dir,
                                         size_t dump_dir_length);
    nncase_dump_flags_t (*compile_options_get_dump_flags)(
        clr_object_handle_t compile_options);
    void (*compile_options_set_dump_flags)(clr_object_handle_t compile_options,
                                           nncase_dump_flags_t dump_flags);
    void (*compile_options_set_quantize_options)(
        clr_object_handle_t compile_options,
        clr_object_handle_t quantize_options);
    clr_object_handle_t (*compile_session_create)(
        clr_object_handle_t target, clr_object_handle_t compile_options);
    clr_object_handle_t (*compile_session_get_compiler)(
        clr_object_handle_t compile_session);
    void (*compiler_initialize)();
    clr_object_handle_t (*compiler_import_module)(clr_object_handle_t compiler,
                                                  clr_object_handle_t stream);
    void (*compiler_compile)(clr_object_handle_t compiler);
    void (*compiler_gencode)(clr_object_handle_t compiler,
                             clr_object_handle_t stream);
    clr_object_handle_t (*datatype_from_typecode)(nncase::typecode_t typecode);
    clr_object_handle_t (*expr_evaluate)(clr_object_handle_t expr,
                                         clr_object_handle_t parameters,
                                         clr_object_handle_t inputs);
    clr_object_handle_t (*function_get_body)(clr_object_handle_t function);
    clr_object_handle_t (*function_get_parameters)(
        clr_object_handle_t function);
    clr_object_handle_t (*ir_module_get_entry)(clr_object_handle_t module);
    void (*luanch_debugger)();
    clr_object_handle_t (*quantize_options_create)();
    void (*quantize_options_set_calibration_dataset)(
        clr_object_handle_t quantize_options, clr_object_handle_t dataset);
    void (*quantize_options_set_calibration_method)(
        clr_object_handle_t quantize_options, nncase_calib_method_t method);
    void (*quantize_options_set_model_quant_mode)(
        clr_object_handle_t quantize_options,
        nncase_model_quant_mode_t model_quant_mode);
    void (*quantize_options_set_quant_type)(
        clr_object_handle_t quantize_options, nncase_quant_type_t quant_type);
    void (*quantize_options_set_w_quant_type)(
        clr_object_handle_t quantize_options, nncase_quant_type_t w_quant_type);
    void (*quantize_options_set_finetune_weights_method)(
        clr_object_handle_t quantize_options,
        nncase_finetune_weights_method_t method);
    void (*quantize_options_set_use_mix_quant)(
        clr_object_handle_t quantize_options, bool use_mix_quant);
    void (*quantize_options_set_quant_scheme)(
        clr_object_handle_t quantize_options, const char *quant_scheme,
        size_t quant_scheme_length);
    void (*quantize_options_set_export_quant_scheme)(
        clr_object_handle_t quantize_options, bool export_quant_scheme);
    void (*quantize_options_set_export_weight_range_by_channel)(
        clr_object_handle_t quantize_options,
        bool export_weight_range_by_channel);

    void (*compile_options_set_shape_bucket_options)(
        clr_object_handle_t compile_options,
        clr_object_handle_t shape_bucket_options);
    clr_object_handle_t (*shape_bucket_options_create)();
    void (*shape_bucket_options_set_enable)(
        clr_object_handle_t shape_bucket_options, bool enable);
    void (*shape_bucket_options_set_range_info)(
        clr_object_handle_t shape_bucket_options, const char *range_info,
        size_t range_info_size);
    void (*shape_bucket_options_set_segments_count)(
        clr_object_handle_t shape_bucket_options, int segments_count);
    void (*shape_bucket_options_set_fix_var_map)(
        clr_object_handle_t shape_bucket_options, const char *fix_var_map,
        size_t fix_var_map_size);

    clr_object_handle_t (*rtvalue_from_handle)(nncase::value_node *value);
    nncase::value_node *(*rtvalue_get_handle)(clr_object_handle_t rtvalue);
    clr_object_handle_t (*stream_create)(const nncase_stream_mt_t *mt,
                                         void *handle);
    clr_object_handle_t (*target_create)(const char *target_name,
                                         size_t target_name_length);
    bool (*target_exists)(const char *target_name, size_t target_name_length);
} nncase_api_mt_t;

NNCASE_API nncase_api_mt_t *nncase_clr_api();
NNCASE_API int nncase_clr_initialize(const char *root_assembly_path);
NNCASE_API int nncase_clr_uninitialize();
}

DEFINE_ENUM_BITMASK_OPERATORS(nncase_dump_flags_t)

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
            nncase_clr_api()->handle_free(handle);
        }
    }

  private:
    clr_object_handle_t handle_;
};

#define CHECK_CLR(x) x

class clr_object_base {
  public:
    constexpr clr_object_base(std::nullptr_t = nullptr) noexcept
        : obj_(nullptr) {}

    clr_object_base(std::in_place_t, clr_object_ptr ptr) noexcept
        : obj_(std::move(ptr)) {}

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

class array : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    array(nncase_array_element_kind_t kind, const clr_object_handle_t *elements,
          size_t length) {
        obj_ = nncase_clr_api()->array_create(kind, elements, length);
    }

    template <class T = clr_object_base> T at(size_t index) {
        return {std::in_place,
                nncase_clr_api()->array_get_item(obj_.get(), index)};
    }

    size_t length() { return nncase_clr_api()->array_get_length(obj_.get()); }

    template <class T = clr_object_base> std::vector<T> to_vector() {
        std::vector<T> vector(length());
        for (size_t i = 0; i < vector.size(); i++) {
            vector[i] = at<T>(i);
        }
        return vector;
    }
};

class calibration_dataset_provider : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    calibration_dataset_provider(array dataset, size_t samples_count,
                                 array fn_params) {
        obj_ = nncase_clr_api()->calibration_dataset_provider_create(
            dataset.get(), samples_count, fn_params.get());
    }
};

class quantize_options : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    quantize_options() { obj_ = nncase_clr_api()->quantize_options_create(); }

    calibration_dataset_provider calibration_dataset() { return nullptr; }
    void calibration_dataset(const calibration_dataset_provider &value) {
        nncase_clr_api()->quantize_options_set_calibration_dataset(obj_.get(),
                                                                   value.get());
    }

    nncase_model_quant_mode_t model_quant_mode() { return nncase_mqm_no_quant; }
    void model_quant_mode(nncase_model_quant_mode_t value) {
        nncase_clr_api()->quantize_options_set_model_quant_mode(obj_.get(),
                                                                value);
    }

    nncase_calib_method_t calibrate_method() { return nncase_calib_noclip; }
    void calibrate_method(nncase_calib_method_t value) {
        nncase_clr_api()->quantize_options_set_calibration_method(obj_.get(),
                                                                  value);
    }

    nncase_quant_type_t quant_type() { return nncase_qt_uint8; }
    void quant_type(nncase_quant_type_t value) {
        nncase_clr_api()->quantize_options_set_quant_type(obj_.get(), value);
    }

    nncase_quant_type_t w_quant_type() { return nncase_qt_uint8; }
    void w_quant_type(nncase_quant_type_t value) {
        nncase_clr_api()->quantize_options_set_w_quant_type(obj_.get(), value);
    }

    nncase_finetune_weights_method_t finetune_weights_method() {
        return nncase_no_finetune_weights;
    }
    void finetune_weights_method(nncase_finetune_weights_method_t value) {
        nncase_clr_api()->quantize_options_set_finetune_weights_method(
            obj_.get(), value);
    }

    bool use_mix_quant() { return false; }
    void use_mix_quant(bool value) {
        nncase_clr_api()->quantize_options_set_use_mix_quant(obj_.get(), value);
    }

    std::string quant_scheme() { return ""; }
    void quant_scheme(std::string_view value) {
        nncase_clr_api()->quantize_options_set_quant_scheme(
            obj_.get(), value.data(), value.length());
    }

    bool export_quant_scheme() { return false; }
    void export_quant_scheme(bool value) {
        nncase_clr_api()->quantize_options_set_export_quant_scheme(obj_.get(),
                                                                   value);
    }

    bool export_weight_range_by_channel() { return false; }
    void export_weight_range_by_channel(bool value) {
        nncase_clr_api()->quantize_options_set_export_weight_range_by_channel(
            obj_.get(), value);
    }
};

class shape_bucket_options : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    shape_bucket_options() { obj_ = nncase_clr_api()->shape_bucket_options_create(); }

    bool enable() { return false; }
    void enable(bool value) {
        nncase_clr_api()->shape_bucket_options_set_enable(obj_.get(), value);
    }

    std::map<std::string, std::tuple<int, int>> range_info() { return {}; }
    void range_info(std::map<std::string, std::tuple<int, int>> value) {
        char buf[sizeof(value)];
        memcpy(buf, &value, sizeof(value));
        nncase_clr_api()->shape_bucket_options_set_range_info(obj_.get(), buf, sizeof(value));
    }

    int segments_count() { return 2; }
    void segments_count(int value) {
        nncase_clr_api()->shape_bucket_options_set_segments_count(obj_.get(), value);
    }

    std::map<std::string, int> fix_var_map() { return {}; }
    void fix_var_map(std::map<std::string, int> value) {
        char buf[sizeof(value)];
        memcpy(buf, &value, sizeof(value));
        nncase_clr_api()->shape_bucket_options_set_fix_var_map(obj_.get(), buf, sizeof(value));
    }
};

class cstream : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    cstream(const nncase_stream_mt_t *mt, void *handle) {
        obj_ = nncase_clr_api()->stream_create(mt, handle);
    }
};

class compile_options : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    compile_options() { obj_ = nncase_clr_api()->compile_options_create(); }

    std::string input_format() { return "cpu"; }
    void input_format(std::string_view value) {
        nncase_clr_api()->compile_options_set_input_format(
            obj_.get(), value.data(), value.length());
    }

    std::string dump_dir() { return "cpu"; }
    void dump_dir(std::string_view value) {
        nncase_clr_api()->compile_options_set_dump_dir(obj_.get(), value.data(),
                                                       value.length());
    }

    nncase_dump_flags_t dump_flags() {
        return nncase_clr_api()->compile_options_get_dump_flags(obj_.get());
    }
    void dump_flags(nncase_dump_flags_t value) {
        nncase_clr_api()->compile_options_set_dump_flags(obj_.get(), value);
    }

    clr::quantize_options quantize_options() { return nullptr; }
    void quantize_options(const clr::quantize_options &value) {
        nncase_clr_api()->compile_options_set_quantize_options(obj_.get(),
                                                               value.get());
    }

    clr::shape_bucket_options shape_bucket_options() { return nullptr; }
    void shape_bucket_options(const clr::shape_bucket_options &value) {
        nncase_clr_api()->compile_options_set_shape_bucket_options(obj_.get(),
                                                                   value.get());
    }
};

class target : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    static bool exists(std::string_view name) {
        return nncase_clr_api()->target_exists(name.data(), name.length());
    }

    target(std::string_view name) {
        obj_ = nncase_clr_api()->target_create(name.data(), name.length());
    }
};

class rtvalue : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    rtvalue(nncase::value_t value) {
        obj_ = nncase_clr_api()->rtvalue_from_handle(value.get());
    }

    value_t to_value() const {
        auto ptr = nncase_clr_api()->rtvalue_get_handle(obj_.get());
        return ptr;
    }
};

class expr : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    rtvalue evaluate(const array &params, const array &inputs) {
        return {std::in_place, nncase_clr_api()->expr_evaluate(
                                   get(), params.get(), inputs.get())};
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
        return {std::in_place, nncase_clr_api()->function_get_body(get())};
    }

    array parameters() {
        return {std::in_place,
                nncase_clr_api()->function_get_parameters(get())};
    }
};

class ir_module : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    function entry() {
        return {std::in_place, nncase_clr_api()->ir_module_get_entry(get())};
    }
};

class compiler : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    ir_module import_module(cstream &stream) {
        return {std::in_place,
                nncase_clr_api()->compiler_import_module(get(), stream.get())};
    }

    void compile() { nncase_clr_api()->compiler_compile(obj_.get()); }
    void gencode(cstream &stream) {
        nncase_clr_api()->compiler_gencode(obj_.get(), stream.get());
    }
};

class compile_session : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    compile_session(const target &target, const compile_options &options) {
        obj_ = nncase_clr_api()->compile_session_create(target.get(),
                                                        options.get());
    }

    clr::compiler compiler() {
        return {std::in_place,
                nncase_clr_api()->compile_session_get_compiler(obj_.get())};
    }
};
} // namespace nncase::clr
