/* Copyright 2019 Canaan Inc.
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
#include "ProgressBar.hpp"
#include "modes.h"
#include "registry.h"
#include "targets/cpu/target.h"
#include "targets/k210/target.h"
#include "targets/target.h"
#include <codegen/codegen.h>
#include <data/dataset.h>
#include <fstream>
#include <importer/importer.h>
#include <io_utils.h>
#include <ir/evaluator.h>
#include <ir/quantizer.h>
#include <scheduler/scheduler.h>
#include <string_view>

#define EVAL 0

using namespace clipp;
using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace nncase::codegen;
using namespace nncase::data;
using namespace nncase::scheduler;
using namespace nncase::transforms;

namespace
{
void dump_graph(graph &graph, std::ostream &output)
{
    graph.assign_names();

    output << "digraph \"graph\" {\n";
    output << "node [shape=\"record\"]\n";

    for (auto &&node : graph.nodes())
        output << "\"" << node->name() << "\" [label=\"{" << node_opcode_names(node->runtime_opcode()) << "}\"]\n";

    for (auto &&node : graph.nodes())
    {
        for (auto &&out : node->outputs())
        {
            auto shape = to_string(out.shape());
            for (auto &&conn : out.connections())
            {
                output << "\"" << node->name() << "\"->\"" << conn->owner().name() << "\" [label=\"" << shape << "\"]\n";
            }
        }
    }

    output << "}" << std::endl;
}

void dump_graph(const compile_options &options, graph &graph, std::string_view prefix)
{
    if (options.dump_ir)
    {
        std::ofstream file("ir_" + std::string(prefix) + ".dot", std::ios::out);
        dump_graph(graph, file);
    }
}

void transform_graph(graph &graph, target &target, xtl::span<std::unique_ptr<transform>> transforms)
{
    std::vector<transform *> transform_refs;
    std::transform(std::begin(transforms), std::end(transforms), std::back_inserter(transform_refs), [](auto &&t) { return t.get(); });
    transform_graph(graph, target, transform_refs);
}

#define SCHEDULE_IMPL()                                               \
    std::vector<std::unique_ptr<memory_allocator>> allocator_holder;  \
    std::unordered_map<memory_type_t, memory_allocator *> allocators; \
    target.fill_allocators(allocators, allocator_holder);             \
    allocation_context alloc_ctx(allocators);                         \
    std::vector<node *> compute_sequence;                             \
    std::cout << "  Plan buffers..." << std::endl;                    \
    schedule(graph.outputs(), alloc_ctx, compute_sequence, options.max_solve_secs);

#define EVAL_IMPL()                                                 \
    SCHEDULE_IMPL();                                                \
    evaluate_context eval_ctx(allocators, alloc_ctx.allocations()); \
    evaluator eval(eval_ctx, compute_sequence);

std::unique_ptr<target> create_target(const compile_options &options)
{
    target_options t_options;
    if (options.input_type == "default")
        t_options.input_type = options.inference_type;
    else
        t_options.input_type = options.input_type;

    if (options.output_format == "kmodel")
    {
        if (options.target == "k210")
            return std::make_unique<k210_target>(t_options);
        else if (options.target == "cpu")
            return std::make_unique<cpu_target>(t_options);
        else
            throw std::invalid_argument("Invalid target: " + options.target);
    }
    else
    {
        throw std::invalid_argument("Invalid output format: " + options.output_format);
    }
}

graph import(const compile_options &options)
{
    auto model = read_file(options.input_filename);

    if (options.input_format == "tflite")
        return import_tflite(model);
    else if (options.input_format == "paddle")
        return import_paddle(model, boost::filesystem::path(options.input_filename).parent_path());
    else if (options.input_format == "caffe")
        return import_caffe(model);
    else
        throw std::invalid_argument("Invalid input format: " + options.input_format);
}

void optimize_pass1(const compile_options &options, target &target, graph &graph)
{
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_default_transforms(transforms);
    target.add_optimize1_transforms(transforms);
    transform_graph(graph, target, transforms);
    dump_graph(options, graph, "optimize_1");
}

void optimize_pass2(const compile_options &options, target &target, graph &graph)
{
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_default_transforms(transforms);
    target.add_optimize2_transforms(transforms);
    transform_graph(graph, target, transforms);
    dump_graph(options, graph, "optimize_2");
}

void add_quantization_checkpoints(const compile_options &options, target &target, graph &graph)
{
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_quantization_checkpoint_transforms(transforms);
    transform_graph(graph, target, transforms);
    dump_graph(options, graph, "before_quant");
}

void quantize_graph(const compile_options &options, target &target, graph &graph, quantizer &quantizer)
{
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_default_transforms(transforms);
    target.add_quantization_transforms(quantizer, transforms);
    transform_graph(graph, target, transforms);
    dump_graph(options, graph, "after_quant");
}

void get_quantization_ranges(target &target, graph &graph, quantizer *quantizer, const compile_options &options)
{
    EVAL_IMPL();

    assert(graph.inputs().size() == 1);
    std::unique_ptr<dataset> ds;
    if (options.dataset_format == "image")
        ds = std::make_unique<image_dataset>(options.dataset, graph.inputs()[0]->output().shape(), options.input_mean, options.input_std);
    else if (options.dataset_format == "raw")
        ds = std::make_unique<raw_dataset>(options.dataset, graph.inputs()[0]->output().shape(), options.input_mean, options.input_std);
    else
        throw std::runtime_error("Invalid dataset format: " + options.dataset_format);
    int i = 0;
    std::cout << "  Run calibration..." << std::endl;

    ProgressBar progress_bar(ds->total_size(), 50);
    progress_bar.display();

    // tell the bar to finish
    for (auto it = ds->begin<float>(); it != ds->end<float>(); ++it)
    {
        auto input = eval.input_at<float>(0);
        auto &tensor = it->tensor;
        std::copy(tensor.begin(), tensor.end(), input.begin());

        eval.evaluate(quantizer, options.use_dataset_as_input_stat);

#if EVAL > 0
        auto output = eval.output_at<float>(0);
        std::ofstream result("q1_" + std::to_string(i++) + ".bin", std::ios::binary | std::ios::out);
        runtime::binary_writer writer(result);
        writer.write_array<float>(output);
#endif
        ++progress_bar;
        progress_bar.display();
    }

    progress_bar.done();
}

template <class T>
void run_quantized_graph_impl(evaluator &eval, dataset &ds)
{
    ProgressBar progress_bar(ds.total_size(), 50);
    progress_bar.display();

    int i = 0;
    // tell the bar to finish
    for (auto it = ds.begin<T>(); it != ds.end<T>(); ++it)
    {
        auto input = eval.input_at<T>(0);
        auto &tensor = it->tensor;
        std::copy(tensor.begin(), tensor.end(), input.begin());

        eval.evaluate();

        auto output = eval.output_at<float>(0);
        std::ofstream result("q2_" + std::to_string(i++) + ".bin", std::ios::binary | std::ios::out);
        runtime::binary_writer writer(result);
        writer.write_array<float>(output);

        ++progress_bar;
        progress_bar.display();
    }

    progress_bar.done();
}

void run_quantized_graph(target &target, graph &graph, const compile_options &options)
{
    EVAL_IMPL();

    assert(graph.inputs().size() == 1);
    std::unique_ptr<dataset> ds;
    if (options.dataset_format == "image")
        ds = std::make_unique<image_dataset>(options.dataset, graph.inputs()[0]->output().shape(), options.input_mean, options.input_std);
    else if (options.dataset_format == "raw")
        ds = std::make_unique<raw_dataset>(options.dataset, graph.inputs()[0]->output().shape(), options.input_mean, options.input_std);
    else
        throw std::runtime_error("Invalid dataset format: " + options.dataset_format);

    std::cout << "  Run quantized graph..." << std::endl;
    switch (graph.inputs()[0]->output().type())
    {
    case dt_float32:
        run_quantized_graph_impl<float>(eval, *ds);
        break;
    case dt_uint8:
        run_quantized_graph_impl<uint8_t>(eval, *ds);
        break;
    default:
        throw std::runtime_error("Unsupported input datatype");
    }
}

void quantize(const compile_options &options, target &target, graph &graph)
{
    // 4.1. Add quantization checkpoints
    std::cout << "  4.1. Add quantization checkpoints..." << std::endl;
    add_quantization_checkpoints(options, target, graph);

    // 4.2 Get activation ranges
    std::cout << "  4.2. Get activation ranges, this may take a while..." << std::endl;
    // quantize
    quantizer quant;
    if (!options.use_dataset_as_input_stat)
    {
        auto min = (0.f - options.input_mean) / options.input_std;
        auto max = (1.f - options.input_mean) / options.input_std;
        value_range<float> input_range { min, max };

        quant.record(graph.inputs()[0]->output(), input_range);
    }

    get_quantization_ranges(target, graph, &quant, options);

    // broadcast quant ranges
    std::unordered_set<ir::node_opcode> opcodes;
    target.add_quantization_broadcast(opcodes);
    quant.broadcast_output(graph, opcodes);

    // 4.3 quantize graph
    std::cout << "  4.3. Quantize graph..." << std::endl;
    quantize_graph(options, target, graph, quant);

#if EVAL > 1
    // 4.4 Run quantized graph
    std::cout << "  4.4. Run quantized graph..." << std::endl;
    run_quantized_graph(target, graph, options);
#endif
}

void gencode(target &target, graph &graph, const compile_options &options)
{
    SCHEDULE_IMPL();

    std::ofstream outfile(options.output_filename, std::ios::binary | std::ios::out);
    if (outfile.bad())
        throw std::runtime_error("Cannot open file for output: " + options.output_filename);

    std::cout << "  Emit code..." << std::endl;
    codegen_context codegen_ctx(outfile, allocators, alloc_ctx.allocations());
    codegen::gencode(codegen_ctx, compute_sequence);
}
}

group compile_options::parser(mode &mode)
{
    // clang-format off
    return (
        command("compile").set(mode, mode::compile),
        "compile" % (
			value("input file", input_filename) % "input file",
			value("output file", output_filename) % "output file",
			required("-i", "--input-format") % "input file format: e.g. tflite, caffe" & value("input format", input_format),
			option("-o", "--output-format") % ("output file format: e.g. kmodel, default is " + output_format) & value("output format", output_format),
			option("-t", "--target") % ("target arch: e.g. cpu, k210, default is " + target) & value("target", target),
			option("--dataset") % "calibration dataset, used in post quantization" & value("dataset path", dataset),
			option("--dataset-format") % ("datset format: e.g. image, raw default is " + dataset_format) & value("dataset format", dataset_format),
			option("--inference-type") % ("inference type: e.g. float, uint8 default is " + inference_type) & value("inference type", inference_type),
			option("--input-mean") % ("input mean, default is " + std::to_string(input_mean)) & value("input mean", input_mean),
			option("--input-std") % ("input std, default is " + std::to_string(input_std)) & value("input std", input_std),
			option("--dump-ir").set(dump_ir) % "dump nncase ir to .dot files",
			option("--input-type").set(input_type) % ("input type: e.g. default, float, uint8, default means equal to inference type") & value("input type", input_type),
			option("--max-allocator-solve-secs") % ("max optimal layout solve time in secs used by allocators, 0 means don't use solver, default is " + std::to_string(max_solve_secs)) & value("max allocator solve secs", max_solve_secs)
		));
    // clang-format on
}

void compile(const compile_options &options)
{
    auto target = create_target(options);

    target->registry_codegen_ops();
    target->registry_evaluator_ops();

    // 1. Import
    std::cout << "1. Import graph..." << std::endl;
    auto graph = import(options);
    dump_graph(options, graph, "import");

    // 2. Optimize Pass 1
    std::cout << "2. Optimize Pass 1..." << std::endl;
    optimize_pass1(options, *target, graph);

    if (options.inference_type == "uint8")
    {
        // 3. Optimize Pass 2
        std::cout << "3. Optimize Pass 2..." << std::endl;
        optimize_pass2(options, *target, graph);
        // 4. Quantize
        std::cout << "4. Quantize..." << std::endl;
        quantize(options, *target, graph);
    }

    // 5. CodeGen
    std::cout << "5. Generate code..." << std::endl;
    gencode(*target, graph, options);
}
