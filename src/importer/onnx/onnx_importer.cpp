/* Copyright 2020 Alexey Chernov <4ernov@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "onnx_importer.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/message.h>
#include <importer/importer.h>
#include <hlir/graph.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

namespace
{
    template <typename Proto>
    bool ParseProtoFromBytes(Proto* proto, const unsigned char* buffer, size_t length)
	{
        // Total bytes hard limit / warning limit are set to 1GB and 512MB
        // respectively.
        ::google::protobuf::io::ArrayInputStream input_stream(buffer, static_cast<int>(length));
        ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
        coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
        return proto->ParseFromCodedStream(&coded_stream);
    }
}

onnx_importer::onnx_importer(xtl::span<const uint8_t> model, hlir::graph &graph)
    : graph_(graph)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!ParseProtoFromBytes(&model_, model.data(), model.size()))
        throw std::runtime_error("Invalid ONNX model");
}

void onnx_importer::import()
{

}

graph nncase::importer::import_onnx(xtl::span<const uint8_t> model)
{
    graph graph;
    onnx_importer(model, graph).import();
    return graph;
}
