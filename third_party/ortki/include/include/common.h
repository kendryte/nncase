#pragma once
#include <core/framework/data_types.h>

namespace ortki
{
#define DEFAULT_OPSET 15

#ifdef _WIN32
#include <intrin.h>
#define ORTKI_API(ret) extern "C" __declspec(dllexport) ret
#else
#define ORTKI_API(ret) extern "C" __attribute__((visibility("default"))) ret
#define __forceinline __attribute__((always_inline)) inline
#endif

    using DataType = ONNX_NAMESPACE::TensorProto_DataType;

#define DATATYPE_TO_T(_datatype, _case_impl, _case_not_impl) \
    switch(_datatype) \
    { \
        _case_not_impl(UNDEFINED); \
        _case_impl(FLOAT, float); \
        _case_impl(UINT8, uint8_t); \
        _case_impl(INT8, int8_t); \
        _case_impl(UINT16, uint16_t); \
        _case_impl(INT16, int16_t); \
        _case_impl(INT32, int32_t); \
        _case_impl(INT64, int64_t); \
        _case_not_impl(STRING); \
        _case_impl(BOOL, bool); \
        _case_impl(FLOAT16, onnxruntime::MLFloat16); \
        _case_impl(DOUBLE, double); \
        _case_impl(UINT32, uint32_t); \
        _case_impl(UINT64, uint64_t); \
        _case_not_impl(COMPLEX64); \
        _case_not_impl(COMPLEX128); \
        _case_impl(BFLOAT16, onnxruntime::BFloat16); \
    }                                                        \
    throw std::runtime_error("Unsupported DataType");

    inline onnxruntime::MLDataType GetDataType(DataType data_type)
    {
#define GET_TYPE(tensor_type, T) \
        case onnx::TensorProto_DataType_##tensor_type: \
            return onnxruntime::DataTypeImpl::GetType<T>();

#define GET_UNIMPL_TYPE(tensor_type) \
        case onnx::TensorProto_DataType_##tensor_type: \
            throw std::runtime_error("Unimplemented input type in OpExecutor::AddInput"); \

        DATATYPE_TO_T(data_type, GET_TYPE, GET_UNIMPL_TYPE)
            //        switch(data_type)
            //        {
            //            GET_UNIMPL_TYPE(UNDEFINED);
            //            GET_TYPE(FLOAT, float);
            //            GET_TYPE(UINT8, uint8_t);
            //            GET_TYPE(INT8, int8_t);
            //            GET_TYPE(UINT16, uint16_t);
            //            GET_TYPE(INT16, int16_t);
            //            GET_TYPE(INT32, int32_t);
            //            GET_TYPE(INT64, int64_t);
            //            GET_UNIMPL_TYPE(STRING);
            //            GET_TYPE(BOOL, bool);
            //            GET_UNIMPL_TYPE(FLOAT16);
            //            GET_TYPE(DOUBLE, double);
            //            GET_TYPE(UINT32, uint32_t);
            //            GET_TYPE(UINT64, uint64_t);
            //            GET_UNIMPL_TYPE(COMPLEX64);
            //            GET_UNIMPL_TYPE(COMPLEX128);
            //            GET_UNIMPL_TYPE(BFLOAT16);
            //        }
            //        throw std::runtime_error("Unsupported DataType");
#undef GET_TYPE
#undef GET_UNIMPL_TYPE
    }
}
