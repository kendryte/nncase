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
//#include "nncase/ir/quantizer.h"
#include <fstream>
#include <magic_enum.hpp>
#include <nncase/codegen/model_builder.h>
#include <nncase/compiler.h>
#include <nncase/data/dataset.h>
#include <nncase/importer/importer.h>
#include <nncase/io_utils.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/evaluator.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/transforms/neutral/add_quant_motion.h>
#include <nncase/transforms/neutral/fold_io_quant_motion.h>
#include <nncase/transforms/neutral/optimize_allocation.h>
#include <nncase/transforms/neutral/optimize_benchmark.h>
#include <nncase/transforms/pass.h>
#include <variant>

using namespace nncase;
using namespace nncase::data;
using namespace nncase::runtime;
using namespace nncase::ir;

namespace
{
calibrate_method to_calibrate_method(std::string name)
{
    if (name == "no_clip")
        return calibrate_method::no_clip;
    if (name == "l2")
        return calibrate_method::l2;
    if (name == "kld_m0")
        return calibrate_method::kld_m0;
    if (name == "kld_m1")
        return calibrate_method::kld_m1;
    if (name == "cdf")
        return calibrate_method::cdf;
    return calibrate_method::no_clip;
}

datatype_t to_datatype_method(std::string name)
{
    if (name == "int8")
        return datatype_t::dt_int8;
    else if (name == "int16")
        return datatype_t::dt_int16;
    else if (name == "int32")
        return datatype_t::dt_int32;
    else if (name == "int64")
        return datatype_t::dt_int64;
    else if (name == "uint8")
        return datatype_t::dt_uint8;
    else if (name == "uint16")
        return datatype_t::dt_uint16;
    else if (name == "uint32")
        return datatype_t::dt_uint32;
    else if (name == "uint64")
        return datatype_t::dt_uint64;
    else if (name == "float16")
        return datatype_t::dt_float16;
    else if (name == "float32")
        return datatype_t::dt_float32;
    else if (name == "float64")
        return datatype_t::dt_float64;
    else if (name == "bfloat16")
        return datatype_t::dt_bfloat16;
    else
        throw std::runtime_error("Unsupported data type");
}

void do_dump_graph(ir::graph &graph, std::ostream &output)
{
    output << "digraph \"graph\" {\n";
    output << "node [shape=\"record\"]\n";

    //for (auto &&node : graph.nodes())
    //{
    //    if (node->runtime_opcode() != ir::op_constant)
    //        output << "\"" << node->name() << "\" [label=\"{" << node->runtime_opcode().name << "}\"]\n";
    //}

    //for (auto &&node : graph.nodes())
    //{
    //    if (node->runtime_opcode() != ir::op_constant)
    //    {
    //        for (auto out : node->outputs())
    //        {
    //            auto shape = std::string(datatype_names(out->type())) + ir::to_string(out->shape());
    //            for (auto &&conn : out->connections())
    //            {
    //                output << "\"" << node->name() << "\"->\"" << conn->owner().name() << "\" [label=\"" << shape << "\"]\n";
    //            }
    //        }
    //    }
    //}

    output << "}" << std::endl;
}

std::string format_size(size_t size)
{
    size_t index = 0;
    double display_size = (double)size;
    std::vector<std::string> size_surfix { "B", "KB", "MB" };
    while (index < size_surfix.size() - 1)
    {
        if (display_size >= 1024)
        {
            display_size /= 1024;
            index++;
        }
        else
        {
            break;
        }
    }

    std::stringstream ss;
    ss << std::setfill(' ') << std::setw(7) << std::fixed << std::setprecision(2) << display_size << ' ' << size_surfix[index] << '\t' << "(" << size << " B)";
    return ss.str();
}

class compiler_impl : public compiler
{
public:
    compiler_impl(const compile_options &options)
        : compile_options_(options)
    {
        graph_.name("main");
        if (!options.dump_dir.empty())
            std::filesystem::create_directories(options.dump_dir);
        set_target(options.target);
    }

    nncase::target &target() noexcept override { return *target_; }

#define BEGIN_IMPORT()                                 \
    std::cout << "1. Import graph..." << std::endl;    \
                                                       \
    importer::import_options imp_options;              \
    imp_options.input_layout = options.input_layout;   \
    imp_options.output_layout = options.output_layout; \
    imp_options.output_arrays = options.output_arrays; \
    input_layout_ = options.input_layout;

#define END_IMPORT()                                                  \
    if (compile_options_.dump_ir)                                     \
    {                                                                 \
        std::ofstream f(compile_options_.dump_dir / "ir_import.dot"); \
        do_dump_graph(graph_, f);                                     \
    }                                                                 \
    dump_graph(graph_, "import");

    void import_tflite(std::span<const uint8_t> model, const import_options &options) override
    {
        BEGIN_IMPORT()
        importer::import_tflite(graph_, model, imp_options);
        END_IMPORT()
    }

    void import_onnx(std::span<const uint8_t> model, const import_options &options) override
    {
        //BEGIN_IMPORT()
        //importer::import_onnx(graph_, model, imp_options);
        //END_IMPORT()
    }

    void use_ptq(ptq_dataset_options options) override
    {
        ptq_options_ = std::move(options);
        use_ptq_ = true;
    }

    void use_ptq(ptq_tensor_options options) override
    {
        ptq_options_ = std::move(options);
        use_ptq_ = true;
    }

    void compile() override
    {
        if (use_ptq_)
        {
            if (compile_options_.input_type == "default")
                compile_options_.input_type = "uint8";
        }
        else
        {
            if (compile_options_.input_type == "default")
                compile_options_.input_type = "float32";
        }

        std::cout << "2. Optimize target independent..." << std::endl;
        optimize_target_independent(graph_);

        std::cout << "3. Optimize target dependent..." << std::endl;
        optimize_target_dependent(graph_, use_ptq_);

        if (use_ptq_)
        {
            std::cout << "4.1. Add quantize annotation..." << std::endl;
            add_quantize_annotation(graph_);

            std::cout << "4.2. Run calibration..." << std::endl;
            auto evaluator = run_calibration(graph_);

            std::cout << "4.3. Quantize graph..." << std::endl;
            quantize_graph(graph_, evaluator);
        }

        std::cout << "5. Optimize target dependent after quantization..." << std::endl;
        optimize_target_dependent_after_quant(graph_);

        std::cout << "6. Optimize modules..." << std::endl;
        optimize_merge_module_regions(graph_);

        if (compile_options_.benchmark_only)
            optimize_benchmark(graph_);
    }

    ir::graph &graph(uint32_t stage) override
    {
        if (stage > 1)
        {
            std::cout << "2. Optimize target independent..." << std::endl;
            optimize_target_independent(graph_);
        }

        if (stage > 2)
        {
            std::cout << "3. Optimize target dependent..." << std::endl;
            optimize_target_dependent(graph_, use_ptq_);
        }

        return graph_;
    }

    void gencode(std::ostream &output) override
    {
        std::cout << "8. Generate code..." << std::endl;
        using namespace nncase::schedule;
        using namespace nncase::codegen;

        scheduler sch(*target_, graph_, graph_.outputs());
        if (compile_options_.dump_ir)
        {
            auto dump_path = compile_options_.dump_dir / "codegen";
            std::filesystem::create_directories(dump_path);
            sch.config_dump(dump_path);
        }

        auto schr = sch.schedule();
        model_builder builder(*target_, schr);
        builder.config_dump(compile_options_.dump_dir, compile_options_.dump_asm);
        auto result = builder.build(output);

        dump_summary(graph_, builder, result);
    }

private:
    void set_target(std::string_view type)
    {
        target_ = plugin_loader::create_target(type);
        target_options_.is_fpga = compile_options_.is_fpga;

        target_->register_evaluator_ops();
    }

    void optimize_target_independent(ir::graph &graph)
    {
        run_passes("target_indep", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { target_->register_target_independent_passes(module_type, pmgr); });
    }

    void optimize_merge_module_regions(ir::graph &graph)
    {
        std::cout << "7.1. Merge module regions..." << std::endl;
        using namespace ir::transforms;

        run_passes("mark_noaction", graph, [&]([[maybe_unused]] const module_type_t &module_type, ir::transforms::pass_manager &pmgr) {
            pmgr.add_pass<make_concat_no_action_pass>();
            pmgr.add_pass<make_bitcast_no_action_pass>();
        });
        graph.merge_module_regions();

        std::cout << "7.2. Optimize buffer fusion..." << std::endl;
        optimize_buffer_fusion(graph);

        std::cout << "7.3. Optimize target dependent after buffer fusion..." << std::endl;
        optimize_target_dependent_after_buffer_fusion(graph);

        dump_graph(graph, "merge_module_regions");
    }

    void optimize_benchmark(ir::graph &graph)
    {
        using namespace ir::transforms;
        run_passes("mark_noaction", graph, [&]([[maybe_unused]] const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { pmgr.add_pass<optimize_benchmark_pass>(); });
        dump_graph(graph, "optimize_benchmark");
    }

    void optimize_buffer_fusion(ir::graph &graph)
    {
        using namespace ir::transforms;

        run_passes("buffer_fusion", graph, [&]([[maybe_unused]] const module_type_t &module_type, ir::transforms::pass_manager &pmgr) {
            pmgr.add_pass<add_copy_to_concat_pass>();
            pmgr.add_pass<add_copy_to_output_pass>();

            transform_pass pass("optimize_copy");
            pass.emplace<remove_exclusive_copy_to_output_transform>();
            pass.emplace<remove_exclusive_copy_to_concat_transform>();
            pmgr.add_pass(std::move(pass));
        });
    }

    void optimize_target_dependent_after_buffer_fusion(ir::graph &graph)
    {
        using namespace ir::transforms;

        run_passes("target_dependent_after_buffer_fusion", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) {
            target_->register_target_dependent_after_buffer_fusion_passes(module_type, pmgr);
            pmgr.add_pass<make_bitcast_no_action_pass>();
        });
    }

    void optimize_target_dependent(ir::graph &graph, bool use_ptq)
    {
        run_passes("target_dep", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { target_->register_target_dependent_passes(module_type, pmgr, use_ptq); });
    }

    void optimize_target_dependent_after_quant(ir::graph &graph)
    {
        run_passes("target_dep_after_quant", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { target_->register_target_dependent_after_quantization_passes(module_type, pmgr); });
    }

    void add_quantize_annotation(ir::graph &graph)
    {
        run_passes("quantize_annotation", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { target_->register_quantize_annotation_passes(module_type, pmgr); });
    }

    void quantize_graph(ir::graph &graph, ir::evaluator &evaluator)
    {
        auto graph_runner = [&](ir::graph &graph) {
            ir::transforms::pass_manager pmgr(graph, *target_);
            auto quant = evaluator.quantizer(graph.module_type());

            if (!compile_options_.use_dataset_as_input_stat)
            {
                float input_mean, input_std;
                std::visit([&](auto &options) {
                    input_mean = options.input_mean;
                    input_std = options.input_std;
                },
                    ptq_options_);

                auto min = (0.f - input_mean) / input_std;
                auto max = (1.f - input_mean) / input_std;
                value_range<float> input_range { min, max };
                quant->set(graph.inputs()[0]->output(), input_range);
                quant->record(graph.inputs()[0]->output(), input_range);
            }

            // broadcast quant ranges
            std::unordered_set<node_opcode> opcodes;
            target_->add_quantization_broadcast(opcodes);
            quant->broadcast_output(graph, opcodes);

            ir::transforms::transform_pass p("process i&o node");

            if (use_ptq_)
            {
                if (compile_options_.input_type != "float32")
                    p.emplace<nncase::ir::transforms::add_input_dequantize_transform>(to_datatype_method(compile_options_.input_type));

                if (compile_options_.output_type != "float32")
                    p.emplace<nncase::ir::transforms::add_output_quantize_transform>(to_datatype_method(compile_options_.output_type));
                pmgr.add_pass(std::move(p));
            }

            pmgr.quantizer(quant);
            if (compile_options_.dump_ir)
                pmgr.dump_dir(compile_options_.dump_dir);
            target_->register_quantize_passes(graph.module_type(), pmgr, to_datatype_method(compile_options_.quant_type), to_datatype_method(compile_options_.w_quant_type));
            pmgr.run();
            dump_graph(graph, "quantize");
        };

        graph_runner(graph);
        for (auto &subgraph : graph.subgraphs())
            graph_runner(*subgraph);
    }

    ir::evaluator run_calibration(ir::graph &graph)
    {
        schedule::scheduler sched(*target_, graph, graph.outputs());
        if (compile_options_.dump_ir)
        {
            auto dump_path = compile_options_.dump_dir / "calibration";
            std::filesystem::create_directories(dump_path);
            sched.config_dump(dump_path);
        }

        auto sched_result = sched.schedule(true);
        ir::evaluator evaluator(sched_result);
        auto calib_method = std::visit([](auto &options) { return to_calibrate_method(options.calibrate_method); },
            ptq_options_);

        evaluator.enable_ptq(*target_, calib_method);

        if (graph.inputs().size() != 1)
            throw std::invalid_argument("PTQ only support models that have single 1 input");

        if (ptq_options_.index() == 0)
        {
            auto &options = std::get<ptq_dataset_options>(ptq_options_);
            auto &in_shape = graph.inputs()[0]->output().shape();
            xt::dynamic_shape<size_t> dataset_in_shape(in_shape.begin(), in_shape.end());
            std::unique_ptr<dataset> ds;
            if (options.dataset_format == "image")
                ds = std::make_unique<image_dataset>(options.dataset, dataset_in_shape, input_layout_, options.input_mean, options.input_std);
            else if (options.dataset_format == "raw")
                ds = std::make_unique<raw_dataset>(options.dataset, dataset_in_shape, options.input_mean, options.input_std);
            else
                throw std::runtime_error("Invalid calibration dataset format: " + options.dataset_format);

            auto in_type = graph.inputs()[0]->output().type();
            switch (in_type)
            {
            case dt_float32:
                run_calibration_eval<float>(options, *ds, evaluator);
                break;
            case dt_uint8:
                run_calibration_eval<uint8_t>(options, *ds, evaluator);
                break;
            case dt_int8:
                run_calibration_eval<int8_t>(options, *ds, evaluator);
                break;
            default:
                throw std::runtime_error("Unsupported input datatype: " + std::string(datatype_names(in_type)));
            }
        }
        else
        {
            auto &options = std::get<ptq_tensor_options>(ptq_options_);
            run_calibration_eval(options, evaluator);
        }

        return evaluator;
    }

    template <class T>
    void run_calibration_eval(ptq_dataset_options &options, dataset &dataset, ir::evaluator &evaluator)
    {
        const size_t max_stages = options.calibrate_method == "no_clip" ? 1 : 2;
        for (size_t stage = 0; stage < max_stages; stage++)
        {
            if (stage == 0)
            {
                std::cout << "4.2.1 Collecting ranges..." << std::endl;
            }
            else
            {
                std::cout << "4.2.2 Collecting distribution..." << std::endl;
                evaluator.begin_collect_distribution();
            }

            size_t i = 0;
            for (auto it = dataset.begin<T>(); it != dataset.end<T>(); ++it)
            {
                auto input_buffer = evaluator.input_at(0).buffer();
                auto &tensor = it->tensor;
                std::memcpy(input_buffer.data(), tensor.data(), input_buffer.size_bytes());

                evaluator.evaluate();
                evaluator.end_sample();
                if (options.progress)
                    options.progress(i++, dataset.total_size());
            }

            if (stage == 1)
            {
                std::cout << "4.2.3. Find optimal quantization ranges..." << std::endl;
                evaluator.end_collect_distribution(options.progress);
            }
        }
    }

    void run_calibration_eval(ptq_tensor_options &options, ir::evaluator &evaluator)
    {
        const size_t max_stages = options.calibrate_method == "no_clip" ? 1 : 2;
        for (size_t stage = 0; stage < max_stages; stage++)
        {
            if (stage == 0)
            {
                std::cout << "4.2.1 Collecting ranges..." << std::endl;
            }
            else
            {
                std::cout << "4.2.2 Collecting distribution..." << std::endl;
                evaluator.begin_collect_distribution();
            }

            for (size_t i = 0; i < options.samples_count; i++)
            {
                auto input_buffer = evaluator.input_at(0).buffer();
                std::memcpy(input_buffer.data(), options.tensor_data.data() + i * input_buffer.size_bytes(), input_buffer.size_bytes());

                evaluator.evaluate();
                evaluator.end_sample();
                if (options.progress)
                    options.progress(i++, options.samples_count);
            }

            if (stage == 1)
            {
                std::cout << "4.2.3. Find optimal quantization ranges..." << std::endl;
                evaluator.end_collect_distribution(options.progress);
            }
        }
    }

    template <class Callable>
    void run_passes(std::string_view name, ir::graph &root_graph, Callable &&register_passes)
    {
        auto graph_runner = [&](ir::graph &graph) {
            ir::transforms::pass_manager pmgr(graph, *target_);
            if (compile_options_.dump_ir)
                pmgr.dump_dir(compile_options_.dump_dir);
            register_passes(graph.module_type(), pmgr);
            pmgr.run();
            dump_graph(graph, name);
        };

        for (auto graph : root_graph.reachable_graphs())
            graph_runner(*graph);
    }

    void dump_graph(ir::graph &graph, std::string_view prefix)
    {
        if (compile_options_.dump_ir)
        {
            auto dump_path = compile_options_.dump_dir / ("ir_" + std::string(prefix));
            std::filesystem::create_directories(dump_path);
            graph.assign_names();
            ir::dump_graph(graph, dump_path);

            for (auto &subgraph : graph.subgraphs())
                dump_graph(*subgraph, prefix);
        }
    }

    void dump_summary(ir::graph &graph, codegen::model_builder &mod_builder, codegen::build_model_result build_result)
    {
        std::cout << "\nSUMMARY" << std::endl;
        std::cout << "INPUTS" << std::endl;
        graph.assign_names();
        size_t i = 0;
        for (auto &in : graph.inputs())
            std::cout << i++ << "\t" << in->name() << "\t" << datatype_names(in->output().type()) << ir::to_string(in->output().shape()) << std::endl;

        std::cout << "OUTPUTS" << std::endl;
        i = 0;
        for (auto &out : graph.outputs())
            std::cout << i++ << "\t" << out->name() << "\t" << datatype_names(out->input().type()) << ir::to_string(out->input().shape()) << std::endl;
        std::cout << "\nMEMORY USAGES" << std::endl;
        size_t total_usage = 0;
        total_usage += dump_memory_usage(mod_builder, mem_input, ".input");
        total_usage += dump_memory_usage(mod_builder, mem_output, ".output");
        total_usage += dump_memory_usage(mod_builder, mem_data, ".data");
        std::cout << "MODEL"
                  << "\t" << format_size(build_result.model_size) << std::endl;
        total_usage += build_result.model_size;
        std::cout << "TOTAL"
                  << "\t" << format_size(total_usage) << std::endl;

        std::ofstream file(compile_options_.dump_dir / "memory_usage.txt");
        file << "input: " << format_size(mod_builder.max_usage(mem_input)) << std::endl;
        file << "output: " << format_size(mod_builder.max_usage(mem_output)) << std::endl;
        file << "data: " << format_size(mod_builder.max_usage(mem_data)) << std::endl;
        file << "MODEL: " << format_size(build_result.model_size) << std::endl;
        file << "TOTAL: " << format_size(total_usage) << std::endl;
    }

    size_t dump_memory_usage(codegen::model_builder &mod_builder, memory_location_t location, std::string_view name)
    {
        auto usage = mod_builder.max_usage(location);
        std::cout << name << "\t" << format_size(usage) << std::endl;
        return usage;
    }

private:
    ir::graph graph_;
    compile_options compile_options_;
    target_options target_options_;
    std::string input_layout_;
    std::variant<ptq_dataset_options, ptq_tensor_options> ptq_options_;
    bool use_ptq_ = false;
    std::unique_ptr<nncase::target> target_;
};
}

compiler::~compiler()
{
}

std::unique_ptr<compiler> compiler::create(const compile_options &options)
{
    return std::make_unique<compiler_impl>(options);
}
