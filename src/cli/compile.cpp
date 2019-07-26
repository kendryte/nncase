#include "compile.h"
#include "registry.h"
#include "targets/cpu/target.h"
#include "targets/target.h"
#include <codegen/codegen.h>
#include <data/dataset.h>
#include <fstream>
#include <importer/importer.h>
#include <ir/evaluator.h>
#include <ir/quantizer.h>
#include <scheduler/scheduler.h>
#include <string_view>

#define KPU 0

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

#define SCHEDULE_IMPL()                                               \
    std::vector<std::unique_ptr<memory_allocator>> allocator_holder;  \
    std::unordered_map<memory_type_t, memory_allocator *> allocators; \
    target.fill_allocators(allocators, allocator_holder);             \
    allocation_context alloc_ctx(allocators);                         \
    std::vector<node *> compute_sequence;                             \
    schedule(graph.outputs(), alloc_ctx, compute_sequence);

#define EVAL_IMPL()                                                 \
    SCHEDULE_IMPL();                                                \
    evaluate_context eval_ctx(allocators, alloc_ctx.allocations()); \
    evaluator eval(eval_ctx, compute_sequence);

void simulate(target &target, graph &graph, image_dataset &dataset)
{
    std::cout << "====== SIMULATION ======" << std::endl;
    EVAL_IMPL();

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

std::unique_ptr<target> create_target(const compile_options &compile_options)
{
    return std::make_unique<cpu_target>();
}

graph import(const compile_options &compile_options)
{
    auto model = read_file(compile_options.input_filename);
    return import_tflite(model);
}

void optimize_pass1(target &target, graph &graph)
{
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_default_transforms(transforms);
    target.add_optimize1_transforms(transforms);
    transform_graph(graph, transforms);
}

void add_quantization_checkpoints(target &target, graph &graph)
{
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_default_transforms(transforms);
    target.add_quantization_checkpoint_transforms(transforms);
    transform_graph(graph, transforms);
}

void get_quantization_ranges(target &target, graph &graph, quantizer *quantizer, const compile_options &compile_options)
{
    EVAL_IMPL();

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

void quantize(target &target, graph &graph, const compile_options &compile_options)
{
    // 3.1. Add quantization checkpoints
    add_quantization_checkpoints(target, graph);

    // 3.2 Get activation ranges
    // quantize
    quantizer quant;
    quant.record(graph.inputs()[0]->output(), { 0.f, 1.f });
    get_quantization_ranges(target, graph, &quant, compile_options);

    // quantize graph
    std::vector<std::unique_ptr<transforms::transform>> transforms;
    target.add_default_transforms(transforms);
    target.add_quantization_transforms(quant, transforms);

#if KPU > 1
    // simulate
    simulate(graph, dataset);
#endif
}

void gencode(target &target, graph &graph, const compile_options &compile_options)
{
    SCHEDULE_IMPL();

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
        option("--dataset") & value("dataset path", dataset),
        option("--inference-type") & value("inference type", inference_type).doc("inference type, default is " + inference_type));
}

void compile(const compile_options &compile_options)
{
    auto target = create_target(compile_options);

    target->registry_codegen_ops();
    target->registry_evaluator_ops();

    // 1. Import
    auto graph = import(compile_options);

    // 2. Optimize Pass 1
    optimize_pass1(*target, graph);

    // 3. Quantize
    if (compile_options.inference_type == "uint8")
        quantize(*target, graph, compile_options);

    // 4. CodeGen
    gencode(*target, graph, compile_options);
}
