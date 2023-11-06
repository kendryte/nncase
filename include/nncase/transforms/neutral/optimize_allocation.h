/* Copyright 2020 Canaan Inc.
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
#pragma once
#include "../pass.h"

namespace nncase::ir::transforms
{
class NNCASE_API make_concat_no_action_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API make_bitcast_no_action_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API make_slice_no_action_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API add_copy_to_concat_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API add_copy_to_slice_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API add_copy_to_output_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API add_copy_to_bitcast_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API remove_exclusive_copy_to_output_transform : public transform
{
public:
    void process(transform_context &context) override;

protected:
    bool on_try_match(ir::node &node, transform_context &context) override;
};

class NNCASE_API remove_exclusive_copy_to_concat_transform : public transform
{
public:
    void process(transform_context &context) override;

protected:
    bool on_try_match(ir::node &node, transform_context &context) override;
};

class NNCASE_API remove_exclusive_copy_to_bitcast_transform : public transform
{
public:
    void process(transform_context &context) override;

protected:
    bool on_try_match(ir::node &node, transform_context &context) override;
};

class NNCASE_API remove_simple_copy_from_slice_transform : public transform
{
public:
    void process(transform_context &context) override;

protected:
    bool on_try_match(ir::node &node, transform_context &context) override;
};

class NNCASE_API remove_non_simple_copy_from_slice_transform : public transform
{
public:
    void process(transform_context &context) override;

protected:
    bool on_try_match(ir::node &node, transform_context &context) override;
};

class NNCASE_API alias_bitcast_buffer_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API alias_concat_buffer_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};

class NNCASE_API alias_slice_buffer_pass : public graph_pass
{
public:
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;
};
}
