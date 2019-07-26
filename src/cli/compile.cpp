#include "compile.h"
#include <codegen/codegen.h>
#include <data/dataset.h>
#include <fstream>
#include <importer/importer.h>
#include <ir/evaluator.h>
#include <ir/quantizer.h>
#include <scheduler/main_memory_allocator.h>
#include <scheduler/scheduler.h>
#include <string_view>
#include <transforms/neutral/fold_quantize.h>
#include <transforms/neutral/fold_transpose.h>
#include <transforms/neutral/transpose_motion.h>

#define KPU 0

using namespace clipp;
using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace nncase::codegen;
using namespace nncase::data;
using namespace nncase::scheduler;
using namespace nncase::transforms;

void init_codegen_ops();
void init_evaluator_ops();

namespace
{
std::vector<uint8_t> read_file(const std::string &filename)
{
    std::ifstream infile(filename, std::ios::binary | std::ios::in);
    if (infile.bad())
        throw std::runtime_error("Cannot open file: " + filename);

    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(length);
    infile.read(reinterpret_cast<char *>(data.data()), length);
    infile.close();
    return data;
}

void transform_graph(graph &graph, xtl::span<std::unique_ptr<transform>> transforms)
{
    std::vector<transform *> transform_refs;
    std::transform(std::begin(transforms), std::end(transforms), std::back_inserter(transform_refs), [](auto &&t) { return t.get(); });
    transform_graph(graph, transform_refs);
}

void simulate(graph &graph, image_dataset &dataset)
{
    std::cout << "====== SIMULATION ======" << std::endl;

    main_memory_allocator const_allocator;
    main_memory_allocator main_allocator;

    std::unordered_map<memory_type_t, memory_allocator *> allocators {
        { mem_const, &const_allocator },
        { mem_main, &main_allocator }
    };
    allocation_context alloc_ctx(allocators);
    std::vector<node *> compute_sequence;

    schedule(graph.outputs(), alloc_ctx, compute_sequence);

    evaluate_context eval_ctx(allocators, alloc_ctx.allocations());
    evaluator eval(eval_ctx, compute_sequence);

    int i = 0;
    for (auto &&batch : dataset)
    {
        auto input = eval.input_at<float>(0);
        std::copy(batch.begin(), batch.end(), input.begin());

        eval.evaluate();

        auto output = eval.output_at<float>(0);
        std::ofstream result(std::to_string(i) + ".bin", std::ios::binary | std::ios::out);
        runtime::binary_writer writer(result);
        writer.write_array<float>(output);
    }
}

graph import(const compile_options &compile_options)
{
    auto model = read_file(compile_options.input_filename);
    return import_tflite(model);
}

void optimize_pass1(graph &graph)
{
    std::vector<std::unique_ptr<transform>> transforms;
    transforms.emplace_back(new fold_transpose_transform());
    transforms.emplace_back(new transpose_motion_transform());
    transforms.emplace_back(new fold_quantize_transform());
#if KPU > 0
    transforms.emplace_back(new replace_strided_conv2d_transform());
    transforms.emplace_back(new kpu_fake_conv2d_transform());
    transforms.emplace_back(new kpu_fake_pool2d_transform());
    transforms.emplace_back(new kpu_fake_conv_with_pool2d_transform());
#endif
    transform_graph(graph, transforms);
}

void add_quantization_checkpoints(graph &graph)
{
#if KPU > 0
    std::vector<std::unique_ptr<transform>> transforms;
    // add quantization checkpoint
    transforms.emplace_back(new kpu_add_quant_checkpoint_transform());
    transform_graph(graph, transforms);
    transform_graph(graph, transforms);
#endif
}

void get_quantization_ranges(graph &graph, quantizer *quantizer, const compile_options &compile_options)
{
    main_memory_allocator const_allocator;
    main_memory_allocator main_allocator;

    std::unordered_map<memory_type_t, memory_allocator *> allocators {
        { mem_const, &const_allocator },
        { mem_main, &main_allocator }
    };
    allocation_context alloc_ctx(allocators);
    std::vector<node *> compute_sequence;

    schedule(graph.outputs(), alloc_ctx, compute_sequence);

    evaluate_context eval_ctx(allocators, alloc_ctx.allocations());
    evaluator eval(eval_ctx, compute_sequence);

    assert(graph.inputs().size() == 1);
    image_dataset dataset(compile_options.dataset, graph.inputs()[0]->output().shape(), 0, 1);
    int i = 0;
    for (auto &&batch : dataset)
    {
        auto input = eval.input_at<float>(0);
        std::copy(batch.begin(), batch.end(), input.begin());

        eval.evaluate(quantizer);

#if !KPU
        auto output = eval.output_at<float>(0);
        std::ofstream result(std::to_string(i) + ".bin", std::ios::binary | std::ios::out);
        runtime::binary_writer writer(result);
        writer.write_array<float>(output);
#endif
    }
}

void quantize(graph &graph, const compile_options &compile_options)
{
    // 3.1. Add quantization checkpoints
    add_quantization_checkpoints(graph);

    // 3.2 Get activation ranges
#if KPU > 0
    // quantize
    quantizer quant;
    quant.record(graph.inputs()[0]->output(), { 0.f, 1.f });
    get_quantization_ranges(graph, &quant, compile_options);
#else
    get_quantization_ranges(graph, nullptr, compile_options);
#endif

#if KPU > 1
    // quantize graph
    transforms.clear();
    transforms.emplace_back(new kpu_conv2d_transform(quant));
    transforms.emplace_back(new fold_quantize_transform());
    transform_graph(graph, transforms);

    // simulate
    simulate(graph, dataset);
#endif
}

void gencode(graph &graph, const compile_options &compile_options)
{
    main_memory_allocator const_allocator;
    main_memory_allocator main_allocator;

    std::unordered_map<memory_type_t, memory_allocator *> allocators {
        { mem_const, &const_allocator },
        { mem_main, &main_allocator }
    };
    allocation_context alloc_ctx(allocators);
    std::vector<node *> compute_sequence;

    schedule(graph.outputs(), alloc_ctx, compute_sequence);

    std::ofstream outfile(compile_options.output_filename, std::ios::binary | std::ios::out);
    if (outfile.bad())
        throw std::runtime_error("Cannot open file for output: " + compile_options.output_filename);

    codegen_context codegen_ctx(outfile, allocators, alloc_ctx.allocations());
    codegen::gencode(codegen_ctx, compute_sequence);
}
}

group compile_options::parser(mode &mode)
{
    return (
        command("compile").set(mode, mode::compile),
        value("input file", input_filename),
        value("output file", output_filename),
        option("--dataset") & value("dataset path", dataset));
}

void compile(const compile_options &compile_options)
{
    init_codegen_ops();
    init_evaluator_ops();

    // 1. Import
    auto graph = import(compile_options);

    // 2. Optimize Pass 1
    optimize_pass1(graph);

    // 3. Quantize
    quantize(graph, compile_options);

    // 4. CodeGen
    gencode(graph, compile_options);
}
