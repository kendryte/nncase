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

#include "../onnx_importer.h"

#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/constant.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Constant(const NodeProto &node)
{
    assert(node.input().size() == 0);
    assert(node.output().size() == 1);

    const auto &output { node.output()[0] };

	hlir::constant* op { };
	if (const auto value { get_attribute<float>(node, "value_float") })
	{
		op = graph_.emplace<constant>(value.value());
	}
	else if (const auto value { get_attribute<int>(node, "value_int") })
	{
		op = graph_.emplace<constant>(static_cast<uint8_t>(value.value()));
	}
	else if (const auto value { get_attribute<xtl::span<const std::int64_t>>(node, "value_ints") })
	{
		auto&& v { value.value() };
		vector<uint8_t>&& vec { begin(v), end(v) };
		shape_t shape { 1, v.size() };
		op = graph_.emplace<constant>(dt_uint8, move(shape), vec);
	}
	else
	{
		throw runtime_error("Constant field format is not supported.");
	}

	assert(op);

	output_tensors_.emplace(output, &op->output());
}
