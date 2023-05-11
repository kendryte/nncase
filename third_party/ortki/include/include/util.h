#pragma once
#include <vector>
#include <core/framework/tensor_shape.h>
#include <onnx/defs/tensor_proto_util.h>
#include <numeric>

namespace ortki {
    class OrtKITensor;

    inline std::vector<int64_t> GetShapeVector(const onnxruntime::TensorShape& shape) {
        std::vector<int64_t> result;
        const auto dims = shape.GetDims();
        result.resize(dims.size());
        result.assign(dims.cbegin(), dims.cend());
        return result;
    }

    template<typename T, typename OT = T>
    inline std::vector<OT> ToVector(T* v, int size) {
        std::vector<T> vec(size);
        for (int i = 0; i < size; ++i) {
            vec[i] = v[i];
        }
        return vec;
    }

    template<>
    inline std::vector<std::string> ToVector(const char** v, int size) {
        std::vector<std::string> vec;
        for (int i = 0; i < size; ++i) {
            vec.emplace_back(v[i]);
        }
        return vec;
    }

    ONNX_NAMESPACE::TensorProto ToTensor(OrtKITensor* tensor);

    template<typename Container>
    inline size_t ComputeSize(const Container& container) {
        return std::accumulate(container.begin(), container.end(), 1, std::multiplies<size_t>());
    }
}
