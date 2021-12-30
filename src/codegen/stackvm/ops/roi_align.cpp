/* Copyright 2019-2021 Canaan Inc.
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
#include "../module_builder.h"

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::stackvm;
using namespace nncase::ir;

void stackvm_module_builder::emit(roi_align &node, stackvm_op_builder &builder)
{
    auto &input = allocation(node.input());
    auto &rois = allocation(node.rois());
    auto &batch_indices = allocation(node.batch_indices());
    auto &output = allocation(node.output());

    builder.lea_buffer(input);
    builder.lea_buffer(rois);
    builder.lea_buffer(batch_indices);
    builder.lea_buffer(output);

    builder.stshape(0, input.shape);
    builder.stshape(1, output.shape);

    builder.tensor_roi_align_(node.input().type(), 0, 1, node.mode(), node.spatial_scale(), node.sampling_ratio());
}
