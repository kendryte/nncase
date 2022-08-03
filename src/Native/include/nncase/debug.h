#pragma once
#include <nncase/runtime/datatypes.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <nncase/value.h>
#include <sstream>

class dump_manager {
    bool append;
    int count = 1;
    std::string currentOp;
    std::filesystem::path dump_root;

  public:
    void set_current_op(const std::string& op) { currentOp = op; }

    std::string get_current_op() { return currentOp; }

    std::filesystem::path dump_path();

    std::filesystem::path get_dump_root() { return dump_root; }

    std::ofstream get_stream() { return get_stream(dump_path()); }

    std::ofstream get_stream(const std::filesystem::path &path);

    int get_count() { return count; }

    void incr_count() {
        count++;
        append = false;
    }

    void set_append(bool app) { append = app; }

    void set_dump_root(std::string root);

    void dump_op(nncase::runtime::stackvm::tensor_function_t tensor_funct);

    void dump_op(const std::string &op);
};

extern dump_manager _dump_manager;

NNCASE_API void set_dump_root(std::string root);
std::filesystem::path dump_path();
std::string to_str(const nncase::dims_t &shape);
void write_out_shape(const nncase::dims_t &shape);
inline void print_dims(const nncase::dims_t &dims, const std::string &name) {
    std::cout << name << ":";
    std::cout << to_str(dims) << std::endl;
}

template <typename F>
void dump(nncase::value_t value, F &&f,
          const std::filesystem::path &path = dump_path()) {
    auto stream = _dump_manager.get_stream(path);
    if (value.is_a<nncase::tensor>()) {
        auto value_tensor = value.as<nncase::tensor>().unwrap();
        f(stream, value_tensor);
        stream.close();
    } else if (value.is_a<nncase::tuple>()) {
        stream << "tuple" << "\n";
        stream.close();
        auto value_tuple = value.as<nncase::tuple>().unwrap();
        for (auto &field : value_tuple->fields()) {
            dump(field, f, path);
        }
    } else {
        std::cout << "unknown in dump" << std::endl;
        stream << "unknown in dump\n";
        return;
    }
}

template <typename F>
void dump_append(F &&f, const std::filesystem::path &path = dump_path()) {
    auto stream = _dump_manager.get_stream(path);
    f(stream);
    _dump_manager.set_append(true);
    stream.close();
}

template <typename T>
void dump_data(std::ostream &stream, const T *data,
               nncase::tensor value_tensor) {
    //    std::cout << "out_shape:";
    //    for (auto d : value_tensor->shape()) {
    //        std::cout << d << " ";
    //    }
    stream << "type:"
           << std::to_string(to_typecode(value_tensor->dtype()).unwrap())
           << std::endl;
    auto shape = value_tensor->shape();
    stream << "shape:" << to_str(shape);
    auto sum = 1;
    for (auto s : shape) {
        sum *= s;
    }
    for (int i = 0; i < sum; ++i) {
        if constexpr (std::is_same_v<T, nncase::half>)
        {
            auto ptr = IN_CAST(uint16_t, data);
            stream << std::to_string((float)nncase::half::from_raw(ptr[i])) << "\n";
        }
        else
        {
            stream << std::to_string(data[i]) << "\n";
        }
    }
}

void dump_output_impl(nncase::value_t value,
                      const std::filesystem::path &path = dump_path(),
                      bool incr = false);

void dump_output(NNCASE_UNUSED nncase::value_t value);

void dump_input(NNCASE_UNUSED nncase::value_t value,
                NNCASE_UNUSED std::string name);