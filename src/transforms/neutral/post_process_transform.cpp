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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/post_process_transform.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

void post_process_transform::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    for (auto out_node : dup(graph.outputs()))
    {
        if (out_node->input().shape().size() == 4)
        {
            auto old_output = out_node->input().connection();
            if (output_layout_ == "NCHW")
            {
                if (real_outlayout_ == "NHWC")
                {
                    auto tp = graph.emplace<transpose>(old_output->type(), old_output->shape(), axis_t { 0, 3, 1, 2 });
                    auto new_output = graph.emplace<output_node>(tp->output().type(), tp->output().shape());
                    tp->input().connect(*old_output);
                    new_output->input().connect(tp->output());
                }
            }
            else
            {
                if (real_outlayout_ == "NCHW")
                {
                    auto tp = graph.emplace<transpose>(old_output->type(), old_output->shape(), axis_t { 0, 2, 3, 1 });
                    auto new_output = graph.emplace<output_node>(tp->output().type(), tp->output().shape());
                    tp->input().connect(*old_output);
                    new_output->input().connect(tp->output());
                }
            }
        }
        graph.dce();
    }
}
