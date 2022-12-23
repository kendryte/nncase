#ifndef NNCASE_BAREMETAL
#include "dump_manager_impl.h"
#include <filesystem>
#include <nncase/runtime/dump_manager.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/stackvm/opcode.h>

namespace fs = std::filesystem;

using namespace nncase;
using namespace nncase::runtime;

void dump_manager::set_dump_root(std::string root) {
    dump_root_.clear();
    fs::path p(root);
    auto dump_root = (p / "Runtime");
    if (!fs::exists(dump_root) && dump_root != "") {
        fs::create_directory(dump_root);
    }
    dump_root_ = dump_root.string();
    // reset count for each dir
    count_ = 1;
}

void dump_manager::dump_op(
    nncase::runtime::stackvm::tensor_function_t tensor_funct) {
    auto func_str = to_string(tensor_funct);
    dump_op(func_str);
}

void dump_manager::dump_op(const std::string &func_str) {
    set_current_op(func_str);
    dump_append(*this, [&](auto &stream) { stream << func_str << std::endl; });
}

std::string dump_manager::dump_path() {
    auto p = fs::path(dump_root_) /
             (std::to_string(get_count()) + "$" + current_op_);
    return p.string();
}

std::ofstream dump_manager::get_stream(const std::string &path) {
    return append_ ? std::ofstream(path, std::ios_base::app)
                   : std::ofstream(path);
}

void write_out_shape(dump_manager &dump_manager_, const nncase::dims_t &shape) {
    auto path = fs::path(dump_manager_.get_dump_root()) / "!out_shape_list";
    auto f = fs::exists(path) ? std::ofstream(path, std::ios::app)
                              : std::ofstream(path);
    f << dump_manager_.get_current_op() << " :" << to_str(shape);
    f.close();
}

const gsl::byte *force_get_data(nncase::tensor tensor) {
    return tensor->to_host()
        .unwrap()
        ->buffer()
        .as_host()
        .unwrap()
        .map(nncase::runtime::map_read)
        .unwrap()
        .buffer()
        .data();
}

void dump_output_impl(dump_manager &dump_manager_, nncase::value_t value,
                      const std::string &path, bool incr) {
#define RETURN_RESULT(_in_type)                                                \
    if (nncase::runtime::cmp_type<_in_type>(value_tensor->dtype())) {          \
        dump_data(stream, IN_CAST(_in_type, data), value_tensor);              \
        return;                                                                \
    }

    dump(
        dump_manager_, value,
        [incr, &dump_manager_](auto &stream, auto &&value_tensor) {
            auto *data = force_get_data(value_tensor);
            if (incr) {
                write_out_shape(dump_manager_, value_tensor->shape());
            }
            if (value_tensor->dtype().template is_a<nncase::value_type_t>()) {
                return;
            }
            RETURN_RESULT_SELECT(RETURN_RESULT);

            if (value_tensor->dtype()->typecode() == nncase::dt_float16) {
                dump_data(stream, IN_CAST(nncase::half, data), value_tensor);
                return;
            }
            //            std::cout << "unsupported type:"
            //                      << (int)value_tensor->dtype()->typecode() <<
            //                      std::endl;
        },
        path);
    if (incr) {
        dump_manager_.incr_count();
    }
#undef RETURN_RESULT
}

void dump_manager::dump_output(nncase::value_t value) {
    dump_output_impl(*this, value, fs::path(dump_path()).string(), true);
}

void dump_manager::dump_input(nncase::value_t value, std::string name) {
    dump_output_impl(*this, value, fs::path(dump_path() + "$" + name).string(),
                     false);
}
#endif