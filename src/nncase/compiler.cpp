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
#include <fstream>
#include <magic_enum.hpp>
#include <nncase/codegen/codegen.h>
#include <nncase/compiler.h>
#include <nncase/importer/importer.h>
#include <nncase/io_utils.h>
#include <nncase/ir/debug.h>
#include <nncase/transforms/pass.h>

using namespace nncase;

namespace
{
void do_dump_graph(ir::graph &graph, std::ostream &output)
{
    output << "digraph \"graph\" {\n";
    output << "node [shape=\"record\"]\n";

    for (auto &&node : graph.nodes())
        output << "\"" << node->name() << "\" [label=\"{" << node->runtime_opcode().name << "}\"]\n";

    for (auto &&node : graph.nodes())
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

    output << "}" << std::endl;
}

class compiler_impl : public compiler
{
public:
    compiler_impl(const compile_options &options)
        : compile_options_(options)
    {
        if (!options.dump_dir.empty())
            std::filesystem::create_directories(options.dump_dir);
        set_target(options.target);
    }

#define BEGIN_IMPORT()                              \
    std::cout << "1. Import graph..." << std::endl; \
                                                    \
    importer::import_options imp_options;           \
    imp_options.output_arrays = options.output_arrays;

#define END_IMPORT() \
    dump_graph(graph_, "import");

    void import_tflite(std::span<const uint8_t> model, const import_options &options) override
    {
        BEGIN_IMPORT()
        importer::import_tflite(graph_, model, imp_options);
        END_IMPORT()
    }

    void compile() override
    {
        std::cout << "2. Optimize target independent..." << std::endl;
        optimize_target_independent(graph_);

        std::cout << "3. Optimize target dependent..." << std::endl;
        optimize_target_dependent(graph_);
    }

    void gencode(std::ostream &output) override
    {
        std::cout << "4. Generate code..." << std::endl;
        using namespace nncase::schedule;
        using namespace nncase::codegen;

        scheduler sch(*target_, graph_.outputs());
        auto schr = sch.schedule();
        //codegen::generator gen(*target_, schr, compile_options_.dump_dir, compile_options_.dump_asm);
        //gen.gencode(output);

        dump_summary(graph_);
    }

private:
    void set_target(std::string_view type)
    {
        target_ = plugin_loader::create_target(type);

        target_->register_codegen_ops();
        target_->register_evaluator_ops();
    }

    void optimize_target_independent(ir::graph &graph)
    {
        ir::transforms::pass_manager pmgr(graph, *target_);
        if (compile_options_.dump_ir)
            pmgr.dump_dir(compile_options_.dump_dir);
        target_->register_target_independent_passes(pmgr);
        pmgr.run();
        dump_graph(graph, "target_indep");
    }

    void optimize_target_dependent(ir::graph &graph)
    {
        ir::transforms::pass_manager pmgr(graph, *target_);
        if (compile_options_.dump_ir)
            pmgr.dump_dir(compile_options_.dump_dir);
        target_->register_target_dependent_passes(pmgr);
        pmgr.run();
        dump_graph(graph, "target_dep");
    }

    void dump_graph(ir::graph &graph, std::string_view prefix)
    {
        if (compile_options_.dump_ir)
        {
            auto file_path = compile_options_.dump_dir / ("ir_" + std::string(prefix) + ".gml");
            graph.assign_names();

            std::ofstream dot_file(file_path.replace_extension(".dot"), std::ios_base::out);
            do_dump_graph(graph, dot_file);
            dot_file.close();
        }
    }

    void dump_summary(ir::graph &graph)
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
    }

private:
    ir::graph graph_;
    compile_options compile_options_;
    target_options target_options_;
    std::unique_ptr<target> target_;
};
}

compiler::~compiler()
{
}

std::unique_ptr<compiler> compiler::create(const compile_options &options)
{
    return std::make_unique<compiler_impl>(options);
}
