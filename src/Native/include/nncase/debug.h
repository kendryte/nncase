#pragma once
#include <filesystem>
#include <iostream>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <nncase/value.h>
#include <sstream>
#include <fstream>

extern bool append;
extern std::string currentOp;
extern std::filesystem::path dump_root;

NNCASE_API void set_dump_root(std::string root);
std::filesystem::path dump_path();
std::ofstream get_stream(const std::filesystem::path &path = dump_path());

template <typename F>
void dump(nncase::value_t value, F &&f,
          const std::filesystem::path &path = dump_path()) {
    auto stream = get_stream(path);
    if (value.is_a<nncase::tensor>()) {
        auto value_tensor = value.as<nncase::tensor>().unwrap();
        f(stream, value_tensor);
        stream.close();
    } else if (value.is_a<nncase::tuple>()) {
        stream.close();
        auto value_tuple = value.as<nncase::tuple>().unwrap();
        for (auto &field : value_tuple->fields()) {
            dump(field, f);
        }
    } else {
        std::cout << "unknown in dump" << std::endl;
        stream << "unknown in dump\n";
        return;
    }
}

template <typename F>
void dump(F &&f, const std::filesystem::path &path = dump_path()) {
    auto stream = get_stream(path);
    f(stream);
    append = true;
    stream.close();
}

std::string to_str(const nncase::dims_t &shape);
void write_shape(const nncase::dims_t &shape);

template <typename T>
void dump_data(std::ostream &stream, const T *data,
               nncase::tensor value_tensor) {
    //    std::cout << "out_shape:";
    //    for (auto d : value_tensor->shape()) {
    //        std::cout << d << " ";
    //    }
    stream << "data type:"
           << std::to_string(to_typecode(value_tensor->dtype()).unwrap())
           << std::endl;
    auto shape = value_tensor->shape();
    stream << "out_shape:" << to_str(shape);
    auto sum = 1;
    for (auto s : shape) {
        sum *= s;
    }
    for (int i = 0; i < sum; ++i) {
        stream << std::to_string(data[i]) << "\n";
    }
}

void dump_output_impl(nncase::value_t value,
                      const std::filesystem::path &path = dump_path(),
                      bool incr = false);

void dump_output(NNCASE_UNUSED nncase::value_t value);

void dump_input(NNCASE_UNUSED nncase::value_t value,
                NNCASE_UNUSED std::string name);

inline void print_dims(const nncase::dims_t & dims, const std::string &name) {
    std::cout << name << ":";
    for(auto dim : dims) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}