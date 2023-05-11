#pragma once
#include "common.h"
#include "util.h"
#include <core/framework/ort_value.h>
#include <core/framework/tensor.h>

namespace ortki {
    class OrtKITensor {
    public:
        OrtKITensor(OrtValue value) noexcept : _value(std::move(value)) {}

        const OrtValue& value() const noexcept {
            return _value;
        }

        OrtValue& value() noexcept {
            return _value;
        }

        const onnxruntime::Tensor& tensor() const noexcept {
            return _value.Get<onnxruntime::Tensor>();
        }

        onnxruntime::Tensor& tensor() noexcept {
            return *_value.GetMutable<onnxruntime::Tensor>();
        }

        DataType data_type() const noexcept {
            return static_cast<DataType>(tensor().GetElementType());
        }
    private:
        OrtValue _value;
    };

    // be used for create tensor array, OrtValue lifetime managed by OrtKITensor
    struct OrtKITensorSeq {
    public:
        OrtKITensorSeq(std::vector<OrtValue> values) : _values(std::move(values)) {}

        size_t size() const { return _values.size(); }

        const std::vector<OrtValue>& values() const noexcept {
            return _values;
        }

        const OrtValue& operator[](size_t index) const noexcept { return _values[index]; }
        OrtValue& operator[](size_t index) noexcept { return _values[index]; }

        const OrtValue& at(size_t index) const { return _values.at(index); }
        OrtValue& at(size_t index) { return _values.at(index); }

        const onnxruntime::Tensor& tensor(size_t index) const { return _values[index].Get<onnxruntime::Tensor>(); }
        onnxruntime::Tensor& tensor(size_t index) { return *_values[index].GetMutable<onnxruntime::Tensor>(); }

        auto begin() const noexcept { return _values.begin(); }
        auto begin() noexcept { return _values.begin(); }

        auto end() const noexcept { return _values.end(); }
        auto end() noexcept { return _values.end(); }
    private:
        std::vector<OrtValue> _values;
    };
}
