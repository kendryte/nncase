#pragma once
#include <fstream>
#include <iostream>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/dump_manager.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <nncase/value.h>
#include <sstream>

BEGIN_NS_NNCASE_RUNTIME

template <typename F> void dump_append(dump_manager &dump_manager_, F &&f) {
    return dump_append(dump_manager_, f, dump_manager_.dump_path());
}

template <typename F>
void dump_append(dump_manager &dump_manager_, F &&f, const std::string &path) {
    auto stream = dump_manager_.get_stream(path);
    dump_manager_.set_append(true);
    f(stream);
    dump_manager_.set_append(false);
    stream.close();
}

template <typename F>
void dump(dump_manager &dump_manager_, nncase::value_t value, F &&f) {
    auto stream = dump_manager_.get_stream(dump_manager_.dump_path());
    dump(dump_manager_, value, f, stream);
    stream.close();
}

template <typename F>
void dump(dump_manager &dump_manager_, nncase::value_t value, F &&f,
          std::ofstream &stream) {

    if (value.is_a<nncase::tensor>()) {
        auto value_tensor = value.as<nncase::tensor>().unwrap();
        f(stream, value_tensor);
    } else if (value.is_a<nncase::tuple>()) {
        //        stream << "tuple" << "\n";
        auto value_tuple = value.as<nncase::tuple>().unwrap();
        for (auto &field : value_tuple->fields()) {
            dump(dump_manager_, field, f, stream);
        }
    } else {
        std::cout << "unknown in dump" << std::endl;
        stream << "unknown in dump\n";
        return;
    }

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
        if constexpr (std::is_same_v<T, nncase::half>) {
            auto ptr = IN_CAST(uint16_t, data);
            stream << std::to_string((float)nncase::half::from_raw(ptr[i]))
                   << "\n";
        } else {
            stream << std::to_string(data[i]) << "\n";
        }
    }
    stream <<"------------------\n"<< std::endl;
}

END_NS_NNCASE_RUNTIME