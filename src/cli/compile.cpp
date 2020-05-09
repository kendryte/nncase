/* Copyright 2019-2020 Canaan Inc.
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
#include <hlir/ops/conv2d.h>
#include <hlir/quantizer.h>
#include <hlir/visitor.h>
#include <importer/importer.h>
#include <io_utils.h>
#include <llir/evaluator.h>
#include <scheduler/scheduler.h>
#include <string_view>

#define EVAL 0

using namespace clipp;
using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;
using namespace nncase::codegen;
using namespace nncase::data;
using namespace nncase::scheduler;

namespace
{
template <class TGraph>
void dump_graph(TGraph &graph, std::ostream &output)
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

template <class TGraph>
void dump_graph(const compile_options &options, TGraph &graph, std::string_view prefix)
{
    if (options.dump_ir)
    {
        std::ofstream file("ir_" + std::string(prefix) + ".dot", std::ios::out);
        dump_graph(graph, file);
    }
}

#define SCHEDULE_IMPL(g, max_solve_secs)                              \
    std::vector<std::unique_ptr<memory_allocator>> allocator_holder;  \
    std::unordered_map<memory_type_t, memory_allocator *> allocators; \
    target.fill_allocators(allocators, allocator_holder);             \
    allocation_context alloc_ctx(allocators);                         \
    std::vector<llir::node *> compute_sequence;                       \
    std::cout << "  Plan buffers..." << std::endl;                    \
    schedule(g.outputs(), alloc_ctx, compute_sequence, max_solve_secs);

#define EVAL_IMPL(g, max_solve_secs)                                      \
    SCHEDULE_IMPL(g, max_solve_secs);                                     \
    llir::evaluate_context eval_ctx(allocators, alloc_ctx.allocations()); \
    llir::evaluator eval(eval_ctx, compute_sequence);

std::unique_ptr<target> create_target(const compile_options &options)
{
    target_options t_options;
    if (options.input_type == "default")
        t_options.input_type = options.inference_type;
    else
        t_options.input_type = options.input_type;
    t_options.weights_quantize_threshold = options.weights_quantize_threshold;
    t_options.output_quantize_threshold = options.output_quantize_threshold;
    t_options.quantize_binary = options.quantize_binary;
    t_options.inference_type = options.inference_type;

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
    else if (options.input_format == "onnx")
        return import_onnx(model);
    else
        throw std::invalid_argument("Invalid input format: " + options.input_format);
}

void optimize_pass1(const compile_options &options, target &target, graph &graph)
{
    hlir::transforms::pass_manager mgr(graph, target);
    target.optimize_target_independent(mgr);
    mgr.run();
    dump_graph(options, graph, "optimize_1");
}

void optimize_pass2(const compile_options &options, target &target, graph &graph)
{
    hlir::transforms::pass_manager mgr(graph, target);
    target.optimize_target_dependent(mgr);
    mgr.run();
    dump_graph(options, graph, "optimize_2");
}

void add_quantization_checkpoints(const compile_options &options, target &target, graph &graph)
{
    hlir::transforms::pass_manager mgr(graph, target);
    target.add_quantization_checkpoints(mgr);
    mgr.run();
    dump_graph(options, graph, "before_quant");
}

void quantize_graph(const compile_options &options, target &target, graph &graph, quantizer &quantizer)
{
    hlir::transforms::pass_manager mgr(graph, target);
    target.optimize_quantize(quantizer, mgr);
    mgr.run();
    dump_graph(options, graph, "after_quant");

    // warning weights divergence
    for (auto &n : graph.nodes())
    {
        if (auto conv = node_cast<hlir::conv2d>(*n))
        {
            if ((conv->attributes() & node_attr_need_quantize) == 0)
            {
                std::cout << "WARN: " << n->name() << " Fallback to float conv2d due to weights divergence." << std::endl;
            }
        }
    }
}

void run_calibrations(target &target, graph &graph, quantizer *quantizer, const compile_options &options)
{
    hlir_compile_context hc_ctx;
    graph.compile(hc_ctx);
    EVAL_IMPL(hc_ctx.graph, 0);

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

        eval.evaluate(quantizer, &hc_ctx.l_outputs, options.use_dataset_as_input_stat);

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
void run_quantized_graph_impl(llir::evaluator &eval, dataset &ds)
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

void dump_weights_range(hlir::graph &graph)
{
    std::cout << "Dump weights range ..." << std::endl;
    for (auto &n : graph.nodes())
    {
        if (auto conv = node_cast<hlir::conv2d>(*n))
        {
            auto &weights = conv->weights();
            auto range = quantizer::get_range(weights.begin(), weights.end());
            std::cout << n->name() << "{" << range.min << ", " << range.max << "}" << std::endl;
        }
    }
}

void run_quantized_graph(target &target, graph &graph, const compile_options &options)
{
    hlir_compile_context hc_ctx;
    graph.compile(hc_ctx);
    EVAL_IMPL(hc_ctx.graph, options.max_solve_secs);

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

    hlir::calibrate_method cali_method;
    if (options.calibrate_method == "no_clip")
        cali_method = hlir::calibrate_method::no_clip;
    else if (options.calibrate_method == "l2")
        cali_method = hlir::calibrate_method::l2;
    else
        throw std::invalid_argument("Invalid calibrate method: " + options.calibrate_method);

    if (options.use_dataset_as_input_stat)
    {
        for (auto &in : graph.inputs())
        {
            in->output().attributes(in->output().attributes() | hlir::cnctr_attr_need_quantize);
        }
    }

    // quantize
    quantizer quant(cali_method, 2048);
    // 4.2 Get activation ranges
    std::cout << "  4.2. Get activation ranges..." << std::endl;
    run_calibrations(target, graph, &quant, options);

    if (cali_method != hlir::calibrate_method::no_clip)
    {
        // 4.3 Get activation distributions
        std::cout << "  4.3. Get activation distributions..." << std::endl;
        quant.begin_collect_distribution();
        run_calibrations(target, graph, &quant, options);
        std::cout << "  4.4. Find optimal thresholds..." << std::endl;
        {
            ProgressBar progress_bar(quant.histograms_count(), 50);
            progress_bar.display();
            quant.end_collect_distribution([&](size_t i) {
                ++progress_bar;
                progress_bar.display();
            });
        }

        std::cout << std::endl;
    }

    if (!options.use_dataset_as_input_stat)
    {
        auto min = (0.f - options.input_mean) / options.input_std;
        auto max = (1.f - options.input_mean) / options.input_std;
        value_range<float> input_range { min, max };

        quant.record(graph.inputs()[0]->output(), input_range);
    }

    // broadcast quant ranges
    std::unordered_set<hlir::node_opcode> opcodes;
    target.add_quantization_broadcast(opcodes);
    quant.broadcast_output(graph, opcodes);

    // 4.3 quantize graph
    std::cout << "  4.5. Quantize graph..." << std::endl;
    quantize_graph(options, target, graph, quant);

#if EVAL > 1
    // 4.4 Run quantized graph
    std::cout << "  4.6. Run quantized graph..." << std::endl;
    run_quantized_graph(target, graph, options);
#endif
}

void optimize_pass3(const compile_options &options, target &target, llir::graph &graph)
{
    llir::transforms::pass_manager mgr(graph, target);
    target.optimize_llir(mgr);
    mgr.run();
    dump_graph(options, graph, "optimize_3");
}

void gencode(target &target, llir::graph &graph, const compile_options &options)
{
    SCHEDULE_IMPL(graph, options.max_solve_secs);

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
			required("-i", "--input-format") % "input file format: e.g. tflite, caffe, onnx" & value("input format", input_format),
			option("-o", "--output-format") % ("output file format: e.g. kmodel, default is " + output_format) & value("output format", output_format),
			option("-t", "--target") % ("target arch: e.g. cpu, k210, default is " + target) & value("target", target),
			option("--dataset") % "calibration dataset, used in post quantization" & value("dataset path", dataset),
			option("--dataset-format") % ("datset format: e.g. image, raw default is " + dataset_format) & value("dataset format", dataset_format),
			option("--inference-type") % ("inference type: e.g. float, uint8 default is " + inference_type) & value("inference type", inference_type),
			option("--input-mean").set(use_dataset_as_input_stat, false) % ("input mean, default is " + std::to_string(input_mean)) & value("input mean", input_mean),
			option("--input-std").set(use_dataset_as_input_stat, false) % ("input std, default is " + std::to_string(input_std)) & value("input std", input_std),
			option("--dump-ir").set(dump_ir) % "dump nncase ir to .dot files",
            option("--dump-weights-range").set(dump_weights_range) % "dump weights range",
			option("--input-type").set(input_type) % ("input type: e.g. default, float, uint8, default means equal to inference type") & value("input type", input_type),
			option("--max-allocator-solve-secs") % ("max optimal layout solve time in secs used by allocators, 0 means don't use solver, default is " + std::to_string(max_solve_secs)) & value("max allocator solve secs", max_solve_secs),
			option("--calibrate-method") % ("calibrate method: e.g. no_clip, l2, default is " + calibrate_method) & value("calibrate method", calibrate_method),
			option("--weights-quantize-threshold") % ("the threshold to control quantizing op or not according to it's weigths range, default is " + std::to_string(weights_quantize_threshold)) & value("weights quantize threshold", weights_quantize_threshold),
			option("--output-quantize-threshold") % ("the threshold to control quantizing op or not according to it's output size, default is " + std::to_string(output_quantize_threshold)) & value("output quantize threshold", output_quantize_threshold),
            option("--no-quantized-binary").set(quantize_binary, false) % "don't quantize binary ops"
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
    graph.assign_names();
    dump_graph(options, graph, "import");

    // 2. Optimize Pass 1
    std::cout << "2. Optimize Pass 1..." << std::endl;
    optimize_pass1(options, *target, graph);

    if (options.dump_weights_range)
        dump_weights_range(graph);

    if (options.inference_type == "uint8")
    {
        // 3. Optimize Pass 2
        std::cout << "3. Optimize Pass 2..." << std::endl;
        optimize_pass2(options, *target, graph);
        // 4. Quantize
        std::cout << "4. Quantize..." << std::endl;
        quantize(options, *target, graph);
    }

    // 5. Lowering
    std::cout << "5. Lowering..." << std::endl;
    hlir::hlir_compile_context llir;
    graph.compile(llir);
    dump_graph(options, llir.graph, "lowering");

    // 2. Optimize Pass 3
    std::cout << "6. Optimize Pass 3..." << std::endl;
    optimize_pass3(options, *target, llir.graph);

    // 7. CodeGen
    std::cout << "7. Generate code..." << std::endl;
    gencode(*target, llir.graph, options);

    std::cout << "\nSUMMARY" << std::endl;
    std::cout << "INPUTS" << std::endl;
    llir.graph.assign_names();
    size_t i = 0;
    for (auto &in : llir.graph.inputs())
        std::cout << i++ << "\t" << in->name() << "\t" << to_string(in->output().shape()) << std::endl;

    std::cout << "OUTPUTS" << std::endl;
    i = 0;
    for (auto &out : llir.graph.outputs())
        std::cout << i++ << "\t" << out->name() << "\t" << to_string(out->input().shape()) << std::endl;
}
