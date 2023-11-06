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
#include "nncase/ir/quantizer.h"
#include "xtensor/xadapt.hpp"
#include <fstream>
#include <magic_enum.hpp>
#include <nncase/codegen/model_builder.h>
#include <nncase/compiler.h>
#include <nncase/data/dataset.h>
#include <nncase/importer/importer.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/evaluator.h>
#include <nncase/kernels/neutral/neutral_kernels.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/debug.h>
#include <nncase/transforms/neutral/add_quant_motion.h>
#include <nncase/transforms/neutral/optimize_allocation.h>
#include <nncase/transforms/neutral/optimize_benchmark.h>
#include <nncase/transforms/neutral/post_process_transform.h>
#include <nncase/transforms/neutral/pre_process_setting.h>
#include <nncase/transforms/pass.h>
#include <variant>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>

using namespace nncase;
using namespace nncase::data;
using namespace nncase::runtime;
using namespace nncase::ir;

static float dot(float *v1, float *v2, size_t size)
{
    float ret = 0.f;
    for (size_t i = 0; i < size; i++)
    {
        ret += v1[i] * v2[i];
    }
    return ret;
}

static float cosine(float *v1, float *v2, size_t size)
{
    return dot(v1, v2, size) / ((sqrt(dot(v1, v1, size)) * sqrt(dot(v2, v2, size))));
}
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
    if (name == "kld_m2")
        return calibrate_method::kld_m2;
    if (name == "cdf")
        return calibrate_method::cdf;
    if (name == "auto_select")
        return calibrate_method::auto_select;
    return calibrate_method::no_clip;
}

void do_dump_graph(ir::graph &graph, std::ostream &output)
{
    output << "digraph \"graph\" {\n";
    output << "node [shape=\"record\"]\n";

    for (auto &&node : graph.nodes())
    {
        if (node->runtime_opcode() != ir::op_constant)
            output << "\"" << node->name() << "\" [label=\"{" << node->runtime_opcode().name << "}\"]\n";
    }

    for (auto &&node : graph.nodes())
    {
        if (node->runtime_opcode() != ir::op_constant)
        {
            for (auto out : node->outputs())
            {
                auto shape = std::string(datatype_names(out->type())) + ir::to_string(out->shape());
                for (auto &&conn : out->connections())
                {
                    output << "\"" << node->name() << "\"->\"" << conn->owner().name() << "\" [label=\"" << shape << "\"]\n";
                }
            }
        }
    }

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

#define BEGIN_IMPORT()                              \
    std::cout << "1. Import graph..." << std::endl; \
                                                    \
    importer::import_options imp_options;           \
    imp_options.output_arrays = options.output_arrays;

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
        importer::import_tflite(graph_, model, imp_options, real_inlayout_, real_outlayout_);
        END_IMPORT()
    }

    void import_onnx(std::span<const uint8_t> model, const import_options &options) override
    {
        BEGIN_IMPORT()
        importer::import_onnx(graph_, model, imp_options, real_inlayout_, real_outlayout_);
        END_IMPORT()
    }

    void import_caffe(std::span<const uint8_t> model, std::span<const uint8_t> prototxt) override
    {
        std::cout << "1. Import graph..." << std::endl;
        importer::import_caffe(graph_, model, prototxt, real_inlayout_, real_outlayout_);
        END_IMPORT()
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

    void dump_range_options(dump_range_dataset_options options) override
    {
        dump_range_options_ = std::move(options);
    }

    void dump_range_options(dump_range_tensor_options options) override
    {
        dump_range_options_ = std::move(options);
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
        if (compile_options_.preprocess)
        {
            std::cout << "1.1 Pre-process..." << std::endl;
            input_layout_ = compile_options_.input_layout;
            if (!compile_options_.model_layout.empty())
            {
                real_inlayout_ = compile_options_.model_layout;
                real_outlayout_ = compile_options_.model_layout;
            }
            pre_process(graph_, compile_options_);
            post_process(graph_, compile_options_);
        }

        if (compile_options_.dump_import_op_range)
        {
            std::ofstream f;
            auto evaluator_import = run_calibration(graph_, nncase::ir::eval_step::after_import);
            quantizer *eval_import_quantizer = evaluator_import.quantizer(graph_.module_type());
            if (compile_options_.dump_ir)
                f.open(compile_options_.dump_dir / "origin_layer_data_range.txt");
            std::cout << "Collected ranges:" << std::endl;
            for (uint32_t i = 0; i < eval_import_quantizer->ranges_insert_order().size(); i++)
            {
                auto quant_layer_connector = eval_import_quantizer->ranges_insert_order()[i];
                auto ranges_map = eval_import_quantizer->ranges();
                std::cout << "node name: " << quant_layer_connector->owner().name() << "   range min: " << ranges_map[quant_layer_connector].min << "   range max: " << ranges_map[quant_layer_connector].max << std::endl;
                f << "node name: " << quant_layer_connector->owner().name() << "   range min: " << ranges_map[quant_layer_connector].min << "   range max: " << ranges_map[quant_layer_connector].max << std::endl;
            }
            f.close();
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
            auto evaluator = run_calibration(graph_, nncase::ir::eval_step::after_calib);
            quantizer *eval_quantizer = evaluator.quantizer(graph_.module_type());

            std::cout << "4.3. Quantize graph..." << std::endl;
            quantize_graph(graph_, evaluator);

            if (compile_options_.dump_quant_error)
            {
                std::cout << "4.4. Evaluate quantized graph..." << std::endl;
                char target_name[MAX_MODULE_TYPE_LENGTH];
                memset(target_name, '\0', sizeof(target_name));
                char *p = const_cast<char *>(compile_options_.target.c_str());
                strcpy(target_name, p);
                if (strcmp(target_name, "cpu"))
                    graph_.set_module_type(to_module_type(target_name));

                auto quant_evaluator = run_calibration(graph_, nncase::ir::eval_step::after_quant);
                quantizer *quant_eval_quantizer = quant_evaluator.quantizer(graph_.module_type());

                std::cout << "4.5. Summarize accumulated quant error for layer output..." << std::endl;
                std::ofstream f, g;
                if (compile_options_.dump_ir)
                {
                    f.open(compile_options_.dump_dir / "layer_quant_error.txt");
                    g.open(compile_options_.dump_dir / "layer_quant_range.txt");
                }

                bool has_quant_map = false;
                for (uint32_t i = 0; i < quant_eval_quantizer->quant_buffers_insert_order().size(); i++)
                {
                    auto quant_layer_connector = quant_eval_quantizer->quant_buffers_insert_order()[i];
                    if (quant_layer_connector->owner().get_output_connectors_quant_map().size() != 0)
                        has_quant_map = true;
                }
                if (compile_options_.dump_ir)
                    std::filesystem::create_directories(compile_options_.dump_dir / "layer_output_data");
                if (has_quant_map)
                {
                    std::cout << "TRUE!!!!" << std::endl;
                    for (uint32_t i = 0; i < quant_eval_quantizer->quant_buffers_insert_order().size(); i++)
                    {
                        auto quant_layer_connector = quant_eval_quantizer->quant_buffers_insert_order()[i];
                        auto data_size = kernels::detail::compute_size(quant_layer_connector->shape());
                        if (quant_layer_connector->owner().get_output_connectors_quant_map().size() != 0)
                        {
                            std::string layer_name = quant_layer_connector->owner().get_node_name_before_quant();
                            std::cout << "layer name: " << layer_name << "\ncosine: " << cosine(eval_quantizer->output_buffers()[quant_layer_connector->owner().get_output_connectors_quant_map()[quant_layer_connector]].data(), quant_eval_quantizer->output_buffers()[quant_layer_connector].data(), data_size) << std::endl;
                            if (compile_options_.dump_ir)
                            {
                                f << "layer name: " << layer_name << "\ncosine: " << cosine(eval_quantizer->output_buffers()[quant_layer_connector->owner().get_output_connectors_quant_map()[quant_layer_connector]].data(), quant_eval_quantizer->output_buffers()[quant_layer_connector].data(), data_size) << std::endl;
                                g << "layer name: " << layer_name << "\n range:\tbefore:[" << eval_quantizer->ranges()[quant_layer_connector->owner().get_output_connectors_quant_map()[quant_layer_connector]].min << "," << eval_quantizer->ranges()[quant_layer_connector->owner().get_output_connectors_quant_map()[quant_layer_connector]].max << "]\tquant:[" << quant_eval_quantizer->ranges()[quant_layer_connector].min << "," << quant_eval_quantizer->ranges()[quant_layer_connector].max << "]" << std::endl;
                                std::ofstream f_data_before_quant;
                                std::ofstream f_data_after_quant;
                                auto data_dir = compile_options_.dump_dir / "layer_output_data";
                                std::replace(layer_name.begin(), layer_name.end(), '/', '_');
                                f_data_before_quant.open(data_dir / (std::to_string(i) + layer_name + "_before_quant.csv"));
                                f_data_after_quant.open(data_dir / (std::to_string(i) + layer_name + "_after_quant.csv"));

                                std::vector<float> f_data_before_quant_data_v = eval_quantizer->output_buffers()[quant_layer_connector->owner().get_output_connectors_quant_map()[quant_layer_connector]];
                                std::vector<std::size_t> shape = { data_size, 1 };
                                xt::xarray<float> f_data_before_quant_data_arr = xt::adapt(f_data_before_quant_data_v, shape);
                                xt::dump_csv(f_data_before_quant, f_data_before_quant_data_arr);

                                std::vector<float> f_data_after_quant_data_v = quant_eval_quantizer->output_buffers()[quant_layer_connector];
                                xt::xarray<float> f_data_after_quant_data_arr = xt::adapt(f_data_after_quant_data_v, shape);
                                xt::dump_csv(f_data_after_quant, f_data_after_quant_data_arr);

                                f_data_before_quant.close();
                                f_data_after_quant.close();
                            }
                        }
                    }
                }
                else
                {
                    std::cout << "FALSE!!!!" << std::endl;
                    for (uint32_t i = 0; i < quant_eval_quantizer->quant_buffers_insert_order().size(); i++)
                    {
                        auto quant_layer_connector = quant_eval_quantizer->quant_buffers_insert_order()[i];
                        auto data_size = kernels::detail::compute_size(quant_layer_connector->shape());
                        std::string layer_name = quant_layer_connector->owner().name();
                        if (quant_layer_connector->owner().runtime_opcode() != ir::op_pad)
                        {
                            std::cout << "layer name: " << layer_name << "\ncosine: " << cosine(eval_quantizer->output_buffers()[quant_layer_connector].data(), quant_eval_quantizer->output_buffers()[quant_layer_connector].data(), data_size) << std::endl;
                            if (compile_options_.dump_ir)
                            {
                                f << "layer name: " << layer_name << "\ncosine: " << cosine(eval_quantizer->output_buffers()[quant_layer_connector].data(), quant_eval_quantizer->output_buffers()[quant_layer_connector].data(), data_size) << std::endl;
                                g << "layer name: " << layer_name << "\n range:\tbefore:[" << eval_quantizer->ranges()[quant_layer_connector].min << "," << eval_quantizer->ranges()[quant_layer_connector].max << "]\tquant:[" << quant_eval_quantizer->ranges()[quant_layer_connector].min << "," << quant_eval_quantizer->ranges()[quant_layer_connector].max << "]" << std::endl;
                                std::ofstream f_data_before_quant;
                                std::ofstream f_data_after_quant;
                                auto data_dir = compile_options_.dump_dir / "layer_output_data";
                                std::replace(layer_name.begin(), layer_name.end(), '/', '_');
                                f_data_before_quant.open(data_dir / (std::to_string(i) + layer_name + "_before_quant.csv"));
                                f_data_after_quant.open(data_dir / (std::to_string(i) + layer_name + "_after_quant.csv"));

                                std::vector<float> f_data_before_quant_data_v = eval_quantizer->output_buffers()[quant_layer_connector];
                                std::vector<std::size_t> shape = { data_size, 1 };
                                xt::xarray<float> f_data_before_quant_data_arr = xt::adapt(f_data_before_quant_data_v, shape);
                                xt::dump_csv(f_data_before_quant, f_data_before_quant_data_arr);

                                std::vector<float> f_data_after_quant_data_v = quant_eval_quantizer->output_buffers()[quant_layer_connector];
                                xt::xarray<float> f_data_after_quant_data_arr = xt::adapt(f_data_after_quant_data_v, shape);
                                xt::dump_csv(f_data_after_quant, f_data_after_quant_data_arr);

                                f_data_before_quant.close();
                                f_data_after_quant.close();
                            }
                        }
                    }
                }
                if (compile_options_.dump_ir)
                {
                    f.close();
                    g.close();
                }
            }
        }

        std::cout << "5. Optimize target dependent after quantization..." << std::endl;
        graph_.set_module_type(to_module_type("stackvm"));
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

            if (use_ptq_)
            {
                std::cout << "4.1. Add quantize annotation..." << std::endl;
                add_quantize_annotation(graph_);
            }
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
        target_->options().is_fpga = compile_options_.is_fpga;
        target_->register_evaluator_ops();
    }

    void pre_process(ir::graph &graph, compile_options &cmp_options)
    {
        using namespace ir::transforms;
        run_passes("pre_process", graph, [&]([[maybe_unused]] const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { pmgr.add_pass<pre_process_transform>(
                                                                                                                                          cmp_options.mean, cmp_options.std, cmp_options.input_range, cmp_options.input_shape, cmp_options.swapRB, input_layout_, cmp_options.input_type, cmp_options.quant_type, real_inlayout_, cmp_options.letterbox_value); });
    }
    void post_process(ir::graph &graph, compile_options &cmp_options)
    {
        using namespace ir::transforms;
        run_passes("post_process", graph, [&]([[maybe_unused]] const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { pmgr.add_pass<post_process_transform>(
                                                                                                                                           cmp_options.output_layout, real_outlayout_); });
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
            pmgr.add_pass<make_slice_no_action_pass>();
            pmgr.add_pass<make_concat_no_action_pass>();
            pmgr.add_pass<make_bitcast_no_action_pass>(); });
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
            pmgr.add_pass<add_copy_to_slice_pass>();
            pmgr.add_pass<add_copy_to_output_pass>();
            pmgr.add_pass<add_copy_to_bitcast_pass>();

            transform_pass pass("optimize_copy");
            pass.emplace<remove_exclusive_copy_to_output_transform>();
            pass.emplace<remove_simple_copy_from_slice_transform>();
            pass.emplace<remove_non_simple_copy_from_slice_transform>();
            pass.emplace<remove_exclusive_copy_to_concat_transform>();
            pass.emplace<remove_exclusive_copy_to_bitcast_transform>();
            pmgr.add_pass(std::move(pass));
        });
    }

    void optimize_target_dependent_after_buffer_fusion(ir::graph &graph)
    {
        using namespace ir::transforms;

        run_passes("target_dependent_after_buffer_fusion", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) {
            target_->register_target_dependent_after_buffer_fusion_passes(module_type, pmgr);
            pmgr.add_pass<make_bitcast_no_action_pass>(); });
    }

    void optimize_target_dependent(ir::graph &graph, bool use_ptq)
    {
        run_passes("target_dep", graph, [&](const module_type_t &module_type, ir::transforms::pass_manager &pmgr) { target_->register_target_dependent_passes(module_type, pmgr, use_ptq, compile_options_.split_w_to_act); });
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
            if (compile_options_.input_type != "float32" && compile_options_.preprocess == true)
            {
                auto min = compile_options_.input_range[0];
                auto max = compile_options_.input_range[1];
                value_range<float> input_range { min, max };
                quant->set(graph.inputs()[0]->output(), input_range);
                quant->record(graph.inputs()[0]->output(), input_range);
            }
            std::unordered_set<node_opcode> opcodes;
            target_->add_quantization_broadcast(opcodes);
            quant->broadcast_output(graph, opcodes);
            if (compile_options_.output_type != "float32")
                quant->set_model_output_range(graph);
            pmgr.quantizer(quant);

            if (compile_options_.dump_ir)
                pmgr.dump_dir(compile_options_.dump_dir);
            target_->register_quantize_passes(graph.module_type(), pmgr, parse_datatype_str(compile_options_.quant_type), compile_options_.w_quant_type, compile_options_.use_mse_quant_w, parse_datatype_str(compile_options_.output_type), output_quant_params_, compile_options_.output_range);
            pmgr.run();
            dump_graph(graph, "quantize");
        };

        graph_runner(graph);
        for (auto &subgraph : graph.subgraphs())
            graph_runner(*subgraph);
    }

    ir::evaluator run_calibration(ir::graph &graph, eval_step step)
    {
        schedule::scheduler sched(*target_, graph, graph.outputs());
        if (compile_options_.dump_ir)
        {
            auto dump_path = step == nncase::ir::eval_step::after_import ? compile_options_.dump_dir / "after_import" : (step == nncase::ir::eval_step::after_calib ? compile_options_.dump_dir / "after_calibration" : compile_options_.dump_dir / "after_quantize");
            std::filesystem::create_directories(dump_path);
            sched.config_dump(dump_path);
        }

        auto sched_result = sched.schedule(true);
        ir::evaluator evaluator(sched_result);

        if (step != eval_step::after_import)
        {
            auto calib_method = std::visit([](auto &options) { return to_calibrate_method(options.calibrate_method); },
                ptq_options_);
            evaluator.enable_ptq(*target_, calib_method);
        }
        else
        {
            auto calib_method = std::visit([](auto &options) { return to_calibrate_method(options.calibrate_method); },
                dump_range_options_);
            evaluator.enable_ptq(*target_, calib_method);
        }

        if (step != eval_step::after_import)
        {
            if (ptq_options_.index() == 0)
            {
                auto &options = std::get<ptq_dataset_options>(ptq_options_);
                auto &in_shape = graph.inputs()[0]->output().shape();
                xt::dynamic_shape<size_t> dataset_in_shape(in_shape.begin(), in_shape.end());
                std::unique_ptr<dataset> ds;
                if (options.dataset_format == "image")
                    ds = std::make_unique<image_dataset>(options.dataset, dataset_in_shape, compile_options_.input_layout);
                else if (options.dataset_format == "raw")
                    ds = std::make_unique<raw_dataset>(options.dataset, dataset_in_shape);
                else
                    throw std::runtime_error("Invalid calibration dataset format: " + options.dataset_format);

                auto in_type = graph.inputs()[0]->output().type();
                switch (in_type)
                {
                case dt_float32:
                    run_calibration_eval<float, ptq_dataset_options>(options, *ds, evaluator, step);
                    break;
                case dt_uint8:
                    run_calibration_eval<uint8_t, ptq_dataset_options>(options, *ds, evaluator, step);
                    break;
                case dt_int8:
                    run_calibration_eval<int8_t, ptq_dataset_options>(options, *ds, evaluator, step);
                    break;
                default:
                    throw std::runtime_error("Unsupported input datatype: " + std::string(datatype_names(in_type)));
                }
            }
            else
            {
                auto &options = std::get<ptq_tensor_options>(ptq_options_);
                run_calibration_eval<ptq_tensor_options>(options, evaluator, step);
            }
        }
        else
        {
            if (dump_range_options_.index() == 0)
            {
                auto &options = std::get<dump_range_dataset_options>(dump_range_options_);
                auto &in_shape = graph.inputs()[0]->output().shape();
                xt::dynamic_shape<size_t> dataset_in_shape(in_shape.begin(), in_shape.end());
                std::unique_ptr<dataset> ds;
                if (options.dataset_format == "image")
                    ds = std::make_unique<image_dataset>(options.dataset, dataset_in_shape, compile_options_.input_layout);
                else if (options.dataset_format == "raw")
                    ds = std::make_unique<raw_dataset>(options.dataset, dataset_in_shape);
                else
                    throw std::runtime_error("Invalid calibration dataset format: " + options.dataset_format);

                auto in_type = graph.inputs()[0]->output().type();
                switch (in_type)
                {
                case dt_float32:
                    run_calibration_eval<float, dump_range_dataset_options>(options, *ds, evaluator, step);
                    break;
                case dt_uint8:
                    run_calibration_eval<uint8_t, dump_range_dataset_options>(options, *ds, evaluator, step);
                    break;
                case dt_int8:
                    run_calibration_eval<int8_t, dump_range_dataset_options>(options, *ds, evaluator, step);
                    break;
                default:
                    throw std::runtime_error("Unsupported input datatype: " + std::string(datatype_names(in_type)));
                }
            }
            else
            {
                auto &options = std::get<dump_range_tensor_options>(dump_range_options_);
                run_calibration_eval<dump_range_tensor_options>(options, evaluator, step);
            }
        }

        return evaluator;
    }

    template <class T, class TOpt>
    void run_calibration_eval(TOpt &options, dataset &dataset, ir::evaluator &evaluator, eval_step step)
    {
        std::string step_str = step == nncase::ir::eval_step::after_import ? "1" : (step == nncase::ir::eval_step::after_calib ? "4.2" : "4.4");
        const size_t max_stages = options.calibrate_method == "no_clip" ? 1 : 2;

        for (size_t stage = 0; stage < max_stages; stage++)
        {
            if (stage == 0)
            {
                std::cout << step_str + ".1. Collecting ranges..." << std::endl;
            }
            else
            {
                std::cout << step_str + ".2. Collecting distribution..." << std::endl;
                evaluator.begin_collect_distribution();
            }

            size_t i = 0;
            for (auto it = dataset.begin<T>(); it != dataset.end<T>(); ++it)
            {
                auto input_buffer = evaluator.input_at(0).buffer();
                auto &tensor = it->tensor;
                std::memcpy(input_buffer.data(), tensor.data(), input_buffer.size_bytes());

                evaluator.evaluate(step, stage, compile_options_.dump_quant_error);
                evaluator.end_sample();
                if (options.progress)
                    options.progress(i++, dataset.total_size());
            }

            if (stage == 1)
            {
                std::cout << step_str + ".3. Find optimal quantization ranges..." << std::endl;
                evaluator.end_collect_distribution(options.progress);
            }
        }
    }

    template <class TOpt>
    void run_calibration_eval(TOpt &options, ir::evaluator &evaluator, eval_step step)
    {
        std::string step_str = step == nncase::ir::eval_step::after_import ? "1" : (step == nncase::ir::eval_step::after_calib ? "4.2" : "4.4");
        const size_t max_stages = options.calibrate_method == "no_clip" ? 1 : 2;
        for (size_t stage = 0; stage < max_stages; stage++)
        {
            if (stage == 0)
            {
                std::cout << step_str + ".1. Collecting ranges..." << std::endl;
            }
            else
            {
                std::cout << step_str + ".2. Collecting distribution..." << std::endl;
                evaluator.begin_collect_distribution();
            }

            for (size_t i = 0; i < options.samples_count; i++)
            {
                uint32_t input_offset = 0;
                for (uint32_t j = 0; j < evaluator.inputs_size(); j++)
                {
                    auto input_buffer = evaluator.input_at(j).buffer();
                    std::memcpy(input_buffer.data(), options.tensor_data.data() + input_offset + i * input_buffer.size_bytes(), input_buffer.size_bytes());
                    input_offset += (options.samples_count * input_buffer.size_bytes());
                }

                evaluator.evaluate(step, stage, compile_options_.dump_quant_error);
                evaluator.end_sample();
                if (options.progress)
                    options.progress(i++, options.samples_count);
            }

            if (stage == 1)
            {
                std::cout << step_str + ".3. Find optimal quantization ranges..." << std::endl;
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

        std::ofstream file(compile_options_.dump_dir / "kmodel_info.txt");
        if (compile_options_.dump_dir.filename().string() == "ptq" and compile_options_.output_type != "float32")
        {
            file << "\nOUTPUT_QUANT_PARAM" << std::endl;
            file << "scale:      " << output_quant_params_.scale << std::endl;
            file << "zero_point: " << output_quant_params_.zero_point << std::endl;

            std::cout << "\nOUTPUT_QUANT_PARAM" << std::endl;
            std::cout << "scale:      " << output_quant_params_.scale << std::endl;
            std::cout << "zero_point: " << output_quant_params_.zero_point << std::endl;
        }
        file << "\nMEMORY USAGES" << std::endl;
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
    std::string input_layout_;
    std::variant<ptq_dataset_options, ptq_tensor_options> ptq_options_;
    std::variant<dump_range_dataset_options, dump_range_tensor_options> dump_range_options_;
    bool use_ptq_ = false;
    std::unique_ptr<nncase::target> target_;
    std::string real_inlayout_ = "";
    std::string real_outlayout_ = "";
    quant_param_t output_quant_params_ = { 0, 0 };
};
}

compiler::~compiler()
{
}

std::unique_ptr<compiler> compiler::create(const compile_options &options)
{
    return std::make_unique<compiler_impl>(options);
}
