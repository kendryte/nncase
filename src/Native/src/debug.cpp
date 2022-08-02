#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/debug.h>

namespace fs = std::filesystem;

dump_manager _dump_manager;

void dump_manager::set_dump_root(std::string root) {
    dump_root.clear();
    fs::path p(root);
    dump_root = (p / "Runtime");
    // reset count for each dir
    count = 1;
}

void set_dump_root(std::string root) {
    _dump_manager.set_dump_root(root);
}

void dump_manager::dump_op(nncase::runtime::stackvm::tensor_function_t tensor_funct) {
    auto func_str = to_string(tensor_funct);
    dump_op(func_str);
}

void dump_manager::dump_op(const std::string &func_str) {
    set_current_op(func_str);
    dump_append([&](auto &stream) { stream << func_str << std::endl; });
}

fs::path dump_manager::dump_path() {
    auto p = dump_root / (std::to_string(get_count()) + currentOp);
    if (!fs::exists(dump_root) && dump_root != "") {
        fs::create_directory(dump_root);
    }
    return p;
}

fs::path dump_path() {
    return _dump_manager.dump_path();
}

std::ofstream dump_manager::get_stream(const fs::path &path) {
    return append ? std::ofstream(path, std::ios_base::app)
                  : std::ofstream(path);
}

std::string to_str(const nncase::dims_t &shape) {
    std::stringstream stream;
    if (shape.size() == 0) {
        stream << "scalar\n";
    } else {
        for (auto d : shape) {
            stream << std::to_string(d) << " ";
        }
        stream << std::endl;
    }
    return stream.str();
}

void write_shape(const nncase::dims_t &shape) {
    auto path = fs::path(_dump_manager.get_dump_root()) / "0000out_shape_list";
    auto f = fs::exists(path) ? std::ofstream(path, std::ios::app)
                              : std::ofstream(path);
    f << _dump_manager.get_current_op() << " :" << to_str(shape);
    f.close();
}

const gsl::byte* force_get_data(nncase::tensor tensor)
{
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

void dump_output_impl(nncase::value_t value, const fs::path &path, bool incr) {
#define RETURN_RESULT(_in_type)                                                \
    if (nncase::runtime::cmp_type<_in_type>(value_tensor->dtype())) {          \
        dump_data(stream, IN_CAST(_in_type, data), value_tensor);              \
    }

    dump(
        value,
        [incr](auto &stream, auto &&value_tensor) {
            auto *data = force_get_data(value_tensor);
            if (incr) {
                write_shape(value_tensor->shape());
            }

            RETURN_RESULT(bool);
            RETURN_RESULT(int32_t);
            RETURN_RESULT(uint32_t);
            RETURN_RESULT(int64_t);
            RETURN_RESULT(uint64_t);
            RETURN_RESULT(float);
        },
        path);
    if (incr) {
        _dump_manager.incr_count();
    }
}

void dump_output(NNCASE_UNUSED nncase::value_t value) {
    dump_output_impl(value, dump_path(), true);
}

void dump_input(NNCASE_UNUSED nncase::value_t value,
                NNCASE_UNUSED std::string name) {
    dump_output_impl(value, fs::path(dump_path().string() + name), false);
}