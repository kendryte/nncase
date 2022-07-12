#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <nncase/value.h>

inline int a = 1;
inline bool append = false;

inline int get_a() { return a; }
inline void incr_a() {
    a++;
    append = false;
}
inline std::string currentOp;
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path dump_root = "";
inline static void set_dump_root(std::string root) {
    // todo:maybe should change path in pytest
    fs::path p(root);
    dump_root = p / "Runtime";
}

inline static fs::path dump_path() {
    auto p = dump_root / (std::to_string(get_a()) + currentOp);
    if (!fs::exists(dump_root) && dump_root != "") {
        fs::create_directory(dump_root);
    }
    return p;
}

inline std::ofstream get_stream(const fs::path &path = dump_path()) {
    return append ? std::ofstream(path, std::ios_base::app)
                  : std::ofstream(path);
}

template <typename F>
inline void dump(nncase::value_t value, F &&f, const fs::path &path = dump_path()) {
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

template <typename F> inline void dump(F &&f, const fs::path &path = dump_path()) {
    auto stream = get_stream(path);
    f(stream);
    append = true;
    stream.close();
}

inline std::string to_str(const nncase::dims_t &shape) {
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

inline void write_shape(const nncase::dims_t &shape) {
    auto path = dump_root / "9999shape";
    auto f = fs::exists(path) ? std::ofstream(path, std::ios::app) : std::ofstream(path);
    f << currentOp << " :" << to_str(shape);
    f.close();
}

template <typename T>
inline void dump_data(std::ostream &stream, const T *data,
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
inline void dump_output_impl(nncase::value_t value,
                             const fs::path &path = dump_path(),
                             bool incr = false) {
    dump(
        value,
        [incr](auto &stream, auto &&value_tensor) {
            auto *data = value_tensor->to_host()
                             .unwrap()
                             ->buffer()
                             .as_host()
                             .unwrap()
                             .map(nncase::runtime::map_read)
                             .unwrap()
                             .buffer()
                             .data();
        if(incr) {
            write_shape(value_tensor->shape());
        }
#define RETURN_RESULT(_in_type)                                                \
    if (nncase::runtime::cmp_type<_in_type>(value_tensor->dtype())) {          \
        dump_data(stream, IN_CAST(_in_type, data), value_tensor);              \
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
        incr_a();
    }
}

inline void dump_output([[maybe_unused]] nncase::value_t value) {
    dump_output_impl(value, dump_path(), true);
}

inline void dump_input([[maybe_unused]] nncase::value_t value,
                       [[maybe_unused]] std::string name) {
    dump_output_impl(value, dump_path() / name, false);
}