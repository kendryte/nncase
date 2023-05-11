#pragma once
#include <vector>
#include <functional>
#include <core/graph/graph.h>
#include <core/graph/model.h>
#include <core/framework/customregistry.h>
#include <core/framework/run_options.h>
#include "allocator_manager.h"
#include "common.h"
#include "tensor.h"
#include "environment.h"

using namespace onnxruntime;

namespace ortki {
    template<typename T>
    struct SeqTensors {
        void AddTensor(const std::vector<int64_t>& shape0, const std::vector<T>& data0) {
            tensors.push_back(Tensor<T>{shape0, data0});
        }

        template<typename U>
        struct Tensor {
            std::vector<int64_t> shape;
            std::vector<U> data;
        };
        std::vector<Tensor<T>> tensors;
    };

    // Function templates to translate C++ types into ONNX_NAMESPACE::TensorProto_DataTypes
    template<typename T>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType();

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<float>() {
        return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<double>() {
        return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int32_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int64_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT64;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<bool>() {
        return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int8_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int16_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT16;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint8_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint16_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint32_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint64_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<std::string>() {
        return ONNX_NAMESPACE::TensorProto_DataType_STRING;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<MLFloat16>() {
        return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<BFloat16>() {
        return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    }

    template<typename T>
    struct TTypeProto {
        TTypeProto(const std::vector<int64_t>* shape = nullptr) {
            proto.mutable_tensor_type()->set_elem_type(TypeToDataType<T>());

            if (shape) {
                auto mutable_shape = proto.mutable_tensor_type()->mutable_shape();
                for (auto i : *shape) {
                    auto* mutable_dim = mutable_shape->add_dim();
                    if (i != -1)
                        mutable_dim->set_dim_value(i);
                    else
                        mutable_dim->set_dim_param("symbolic");
                }
            }
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    // Variable template for ONNX_NAMESPACE::TensorProto_DataTypes, s_type_proto<float>, etc..
    template<typename T>
    struct TTensorType {
        static const TTypeProto<T> s_type_proto;
    };

    template<typename T>
    const TTypeProto<T> TTensorType<T>::s_type_proto;

#if !defined(DISABLE_SPARSE_TENSORS)

    struct TSparseTensorProto {
        explicit TSparseTensorProto(int32_t dtype, const std::vector<int64_t>* shape = nullptr) {
            proto.mutable_sparse_tensor_type()->set_elem_type(dtype);
            if (shape) {
                auto m_shape = proto.mutable_sparse_tensor_type()->mutable_shape();
                for_each(shape->cbegin(), shape->cend(), [m_shape](int64_t v) {
                    auto* m_dim = m_shape->add_dim();
                    if (v != -1)
                        m_dim->set_dim_value(v);
                    else
                        m_dim->set_dim_param("symbolic");
                    });
            }
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

#endif

    // TypeProto for map<TKey, TVal>
    template<typename TKey, typename TVal>
    struct MTypeProto {
        MTypeProto() {
            proto.mutable_map_type()->set_key_type(TypeToDataType<TKey>());
            proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(
                TypeToDataType<TVal>());
            proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    template<typename TKey, typename TVal>
    struct MMapType {
        static const MTypeProto<TKey, TVal> s_map_type_proto;
    };

    template<typename TKey, typename TVal>
    const MTypeProto<TKey, TVal> MMapType<TKey, TVal>::s_map_type_proto;

    // TypeProto for vector<map<TKey, TVal>>
    template<typename TKey, typename TVal>
    struct VectorOfMapTypeProto {
        VectorOfMapTypeProto() {
            auto* map_type = proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
            map_type->set_key_type(TypeToDataType<TKey>());
            map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(TypeToDataType<TVal>());
            map_type->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    template<typename TKey, typename TVal>
    struct VectorOfMapType {
        static const VectorOfMapTypeProto<TKey, TVal> s_vec_map_type_proto;
    };

    template<typename TKey, typename TVal>
    const VectorOfMapTypeProto<TKey, TVal> VectorOfMapType<TKey, TVal>::s_vec_map_type_proto;

    template<typename ElemType>
    struct SequenceTensorTypeProto {
        SequenceTensorTypeProto() {
            MLDataType dt = DataTypeImpl::GetTensorType<ElemType>();
            const auto* elem_proto = dt->GetTypeProto();
            proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(*elem_proto);
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    template<typename ElemType>
    struct SequenceTensorType {
        static const SequenceTensorTypeProto<ElemType> s_sequence_tensor_type_proto;
    };

    template<typename ElemType>
    const SequenceTensorTypeProto<ElemType> SequenceTensorType<ElemType>::s_sequence_tensor_type_proto;

#if !defined(DISABLE_OPTIONAL_TYPE)

    template<typename ElemType>
    struct OptionalTypeProto {
        OptionalTypeProto(const ONNX_NAMESPACE::TypeProto& type_proto) {
            proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(type_proto);
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

#endif

    class OpExecutor {
    public:
        // Default to the first opset that ORT was available (7).
        // When operators are updated they need to explicitly add tests for the new opset version.
        // This is due to the kernel matching logic. See KernelRegistry::VerifyKernelDef.
        // Additionally, -1 is supported and defaults to the latest known opset.
        //
        // Defaulting to the latest opset version would result in existing operator implementations for non-CPU EPs to
        // lose their test coverage until an implementation for the new version is added.
        //   e.g. there are CPU and GPU implementations for version 1 of an op. both are tested by a single OpTester test.
        //        opset changes from 1 to 2 and CPU implementation gets added. If 'opset_version' is 2 the kernel matching
        //        will find and run the CPU v2 implementation, but will not match the GPU v1 implementation.
        //        OpTester will say it was successful as at least one EP ran, and the GPU implementation of v1 no longer has
        //        test coverage.
        explicit OpExecutor(const char* op, int opset_version = DEFAULT_OPSET, const char* domain = onnxruntime::kOnnxDomain,
            bool verify_output = true)
            : op_(op), domain_(domain), opset_version_(opset_version), verify_output_(verify_output) {
            if (opset_version_ < 0) {
                static int latest_onnx_version =
                    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(
                        ONNX_NAMESPACE::ONNX_DOMAIN).second;
                opset_version_ = latest_onnx_version;
            }
        }

        void SetOutputSize(size_t size)
        {
            output_size_ = size;
        }

        void AddInput(const char* name, OrtKITensor* tensor)
        {
            ONNX_NAMESPACE::TypeProto proto;
            proto.mutable_tensor_type()->set_elem_type(tensor->tensor().GetElementType());
            if (add_shape_to_tensor_data_)
            {
                auto mutable_shape = proto.mutable_tensor_type()->mutable_shape();
                for (auto i : tensor->tensor().Shape().GetDims()) {
                    auto* mutable_dim = mutable_shape->add_dim();
                    if (i != -1)
                        mutable_dim->set_dim_value(i);
                    else
                        mutable_dim->set_dim_param("symbolic");
                }
            }

            input_data_.emplace_back(NodeArg(name, &proto), tensor->value());
        }

        void AddInput(std::string name, OrtKITensor* tensor)
        {
            ONNX_NAMESPACE::TypeProto proto;
            proto.mutable_tensor_type()->set_elem_type(tensor->tensor().GetElementType());
            if (add_shape_to_tensor_data_)
            {
                auto mutable_shape = proto.mutable_tensor_type()->mutable_shape();
                for (auto i : tensor->tensor().Shape().GetDims()) {
                    auto* mutable_dim = mutable_shape->add_dim();
                    if (i != -1)
                        mutable_dim->set_dim_value(i);
                    else
                        mutable_dim->set_dim_param("symbolic");
                }
            }

            input_data_.emplace_back(NodeArg(std::move(name), &proto), tensor->value());
        }

        void AddSeqInput(const char* name, OrtKITensor** tensors, size_t size)
        {
            std::unique_ptr<TensorSeq> ptr;
            ONNX_NAMESPACE::TypeProto proto;

            if (size)
            {
                auto datatype = tensors[0]->tensor().DataType();
                std::vector<Tensor> seq_tensors;
                seq_tensors.reserve(size);
                for (size_t i = 0; i < size; ++i) {
                    auto& src_tensor = tensors[i]->tensor();
                    seq_tensors.emplace_back(datatype, src_tensor.Shape(), src_tensor.MutableDataRaw(), OrtMemoryInfo());
                }

                ptr = std::make_unique<TensorSeq>(datatype);
                ptr->SetElements(std::move(seq_tensors));

                ONNX_NAMESPACE::TypeProto elem_proto;
                elem_proto.mutable_tensor_type()->set_elem_type(tensors[0]->tensor().GetElementType());
                proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(elem_proto);
            }

            OrtValue value;
            auto mltype = DataTypeImpl::GetType<TensorSeq>();

            std::vector<OrtValue> holder;
            holder.reserve(size);
            for (size_t i = 0; i < size; i++)
                holder.emplace_back(tensors[i]->value());

            // nullptr means None OrtValue which we will skip inserting into the feeds
            value.Init(ptr ? ptr.release() : nullptr, mltype, [holder = std::move(holder), mltype](void* p)
                {
                    mltype->GetDeleteFunc()(p);
                });

            input_data_.emplace_back(NodeArg(name, &proto), std::move(value));
        }

        /*
        * Use this API to add an input *edge* to the node/op being tested that won't
        * have any data passed into.
        * Such an edge will have the qualifier OpSchema::Optional in the schema.
        * This is exposed to ensure the op kernel implementations can be tested to handle
        * presence/absence of such optional input edges.
        */
        template<typename T>
        void AddOptionalInputEdge() {
            std::string name;  // empty == input doesn't exist
            input_data_.emplace_back(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue());
        }

        // Generate the reference outputs with the model file
        // void AddReferenceOutputs(const std::string &model_path);

        void AddCustomOpRegistry(std::shared_ptr<CustomRegistry> registry) {
            custom_schema_registries_.push_back(registry->GetOpschemaRegistry());
            custom_session_registries_.push_back(registry);
        }

        template<typename T>
        void AddAttribute(std::string name, T value) {
            // Generate a the proper AddAttribute call for later
            add_attribute_funcs_.emplace_back(
                [name = std::move(name), value = std::move(value)](onnxruntime::Node& node) {
                    node.AddAttribute(name, value);
                });
        }

        enum class ExpectResult {
            kExpectSuccess,
            kExpectFailure
        };

        std::vector<OrtValue> Run(const RunOptions* run_options = nullptr,
            std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
            ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL);

        std::vector<OrtValue> Run(SessionOptions session_options,
            const RunOptions* run_options = nullptr,
            std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
            const Graph::ResolveOptions& resolve_options = {},
            /*out*/ size_t* number_of_pre_packed_weights_counter = nullptr,
            /*out*/ size_t* number_of_shared_pre_packed_weights_counter = nullptr);

        std::vector<OrtValue>
            GetFetches() { return fetches_; }

        std::unique_ptr<onnxruntime::Model>
            BuildGraph(const std::unordered_map<std::string, int>& extra_domain_to_version = {},
                bool allow_released_onnx_opset_only = true);

        // storing p_model as cache
        void SetModelCache(std::shared_ptr<onnxruntime::Model> model) {
            cached_model_ = model;
        }

        std::shared_ptr<onnxruntime::Model> GetModelCache() {
            return cached_model_;
        }

        // clear input/output data, fetches will be cleared in Run()
        void ClearData() {
            input_data_.clear();
            output_data_.clear();
        }

        struct Data {
            onnxruntime::NodeArg def_;
            OrtValue data_;

            Data(onnxruntime::NodeArg def, OrtValue data)
                : def_(std::move(def)),
                data_(std::move(data)) {}
        };

        std::vector<Data>& GetInputData() {
            return input_data_;
        }

        std::vector<Data>& GetOutputData() {
            return output_data_;
        }

        void SetDeterminism(bool use_determinism) {
            use_determinism_ = use_determinism;
        }

        void EnableSharingOfPrePackedWeightsAcrossSessions() {
            add_prepacked_shared_container_to_sessions_ = true;
        }

        size_t GetNumPrePackedWeightsShared() const {
            return prepacked_weights_container_.GetNumberOfElements();
        }

        bool test_allow_released_onnx_opset_only_ = true;

    protected:
        // Set test_allow_released_onnx_opset_only_ to false or override this method and return false
        // if inheriting from OpTester to allow testing of a non-released ONNX opset operator
        virtual bool IsAllowReleasedONNXOpsetsOnlySetForThisTest() const {
            return test_allow_released_onnx_opset_only_;
        }

        virtual void AddNodes(onnxruntime::Graph& graph, std::vector<onnxruntime::NodeArg*>& graph_input_defs,
            std::vector<onnxruntime::NodeArg*>& graph_output_defs,
            std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs);

        void AddInitializers(onnxruntime::Graph& graph);

        void FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
            std::vector<std::string>& output_names);

        void FillFeeds(std::unordered_map<std::string, OrtValue>& feeds);

        template<class SessionType>
        std::vector<OrtValue> ExecuteModel(Model& model,
            SessionType& session_object,
            const RunOptions* run_options,
            const std::unordered_map<std::string, OrtValue>& feeds,
            const std::vector<std::string>& output_names,
            const std::string& provider_type,
            bool allow_released_onnx_opset_only = true);

        const char* op_;
        std::vector<Data> input_data_;
        std::vector<Data> output_data_;
        std::vector<OrtValue> fetches_;

        // for gradient unit tests only
        std::shared_ptr<onnxruntime::Model> cached_model_;

#ifndef NDEBUG
        bool run_called_{};
#endif
    private:
        std::vector<int64_t> GetDimsForProto(gsl::span<const int64_t> dims);

        void AddShapeToTensorData(NodeArg& node_arg, gsl::span<const int64_t> dims,
            const std::vector<std::string>* dim_params);

        void CopyDataToTensor(gsl::span<const gsl::byte> data, Tensor& dst);

        void InitOutput();

        void AllocOutput(Graph& graph);

        void GraphResolve(Graph& graph, const Graph::ResolveOptions& options, bool cache_enabled)
        {
            Status status = Status::OK();
            if (!cache_enabled) {
                if (add_shape_to_tensor_data_) {
                    //if (add_shape_to_tensor_data_ &&
                    //    expect_result == ExpectResult::kExpectFailure) {
                    // capture possible exceptions from shape inference for invalid testcase
                    ORT_TRY{
                        status = graph.Resolve(options);
                    }
                        ORT_CATCH(const std::exception & ex) {
                        ORT_HANDLE_EXCEPTION([&]() {
                            status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
                            });
                    }
                }
                else {
                    status = graph.Resolve(options);
                }

                //                if (!status.IsOK()) {
                //                    if (expect_result == ExpectResult::kExpectFailure) {
                //                        EXPECT_TRUE(!status.IsOK());
                //                        EXPECT_THAT(status.ErrorMessage(),
                //                                    testing::HasSubstr(expected_failure_string));
                //                    } else {
                //                        LOGS_DEFAULT(ERROR) << "Resolve failed with status: "
                //                                            << sstd::cout <<tatus.ErrorMessage();
                //                        EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
                //                    }
                //                }


                if (!status.IsOK()) {
                    std::cout << status.ErrorMessage() << std::endl;
                    throw std::runtime_error("Graph Resolve Filed:" + status.ErrorMessage());
                    //                    return;
                }
            }
        }
    private:
        size_t output_size_ = INT32_MAX;
        const char* domain_;
        int opset_version_;
        bool add_shape_to_tensor_data_ = true;
        int add_symbolic_dim_to_tensor_data_ = -1;
        int num_run_calls_ = 1;
        std::vector<std::function<void(onnxruntime::Node& node)>> add_attribute_funcs_;

        IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
        std::vector<std::shared_ptr<CustomRegistry>> custom_session_registries_;

        bool verify_output_;

        bool use_determinism_ = false;

        bool add_prepacked_shared_container_to_sessions_ = false;

        onnxruntime::PrepackedWeightsContainer prepacked_weights_container_;
    };
}
