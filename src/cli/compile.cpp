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
#include "compile.h"
#include <nncase/compiler.h>
#include <nncase/io_utils.h>

using namespace nncase;
using namespace nncase::cli;

compile_command::compile_command(lyra::cli &cli)
{
    cli.add_argument(lyra::command("compile", [this](const lyra::group &) { this->run(); })
                         .add_argument(lyra::opt(input_format_, "input format").name("-i").name("--input-format").required().help("input format, e.g. tflite"))
                         .add_argument(lyra::opt(target_name_, "target").name("-t").name("--target").required().help("target architecture, e.g. cpu, k210"))
                         .add_argument(lyra::opt(input_type_, "input type").name("--input_type").name("-it").optional().help("post trainning quantize input type, e.g float32|uint8"))
                         .add_argument(lyra::opt(output_type_, "output type").name("--output_type").name("-ot").optional().help("post trainning quantize output type, e.g float32|uint8"))
                         .add_argument(lyra::arg(input_filename_, "input file").required().help("input file"))
                         .add_argument(lyra::arg(output_filename_, "output file").required().help("output file"))
                         .add_argument(lyra::opt(output_arrays_, "output arrays").name("--output-arrays").optional().help("output arrays"))
                         .add_argument(lyra::opt(dataset_, "dataset path").name("--dataset").optional().help("calibration dataset, used in post quantization"))
                         .add_argument(lyra::opt(dataset_format_, "dataset format").name("--dataset-format").optional().help("datset format: e.g. image, raw default is " + dataset_format_))
                         .add_argument(lyra::opt(calibrate_method_, "calibrate method").name("--calibrate-method").optional().help("calibrate method: e.g. no_clip, l2, default is " + calibrate_method_))
                         .add_argument(lyra::opt(is_fpga_).name("--is-fpga").optional().help("use fpga parameters"))
                         .add_argument(lyra::opt(dump_ir_).name("--dump-ir").optional().help("dump ir to .dot"))
                         .add_argument(lyra::opt(dump_asm_).name("--dump-asm").optional().help("dump assembly"))
                         .add_argument(lyra::opt(dump_dir_, "dump directory").name("--dump-dir").optional().help("dump to directory")));
}

void compile_command::run()
{
    compile_options c_options;
    c_options.dump_asm = dump_asm_;
    c_options.dump_ir = dump_ir_;
    c_options.dump_dir = dump_dir_;
    c_options.target = target_name_;
    c_options.is_fpga = is_fpga_;
    c_options.input_type = input_type_;
    c_options.output_type = output_type_;

    import_options i_options;
    std::vector<std::string> output_arrays;
    if (!output_arrays_.empty())
    {
        auto begin = output_arrays_.begin();
        while (true)
        {
            auto it = std::find(begin, output_arrays_.end(), ',');
            if (it != output_arrays_.end())
            {
                output_arrays.emplace_back(begin, it);
                begin = std::next(it);
            }
            else
            {
                output_arrays.emplace_back(begin, output_arrays_.end());
                break;
            }
        }
    }

    i_options.output_arrays = output_arrays;

    auto compiler = nncase::compiler::create(c_options);
    if (input_format_ == "tflite")
    {
        auto file_data = read_file(input_filename_);
        compiler->import_tflite(file_data, i_options);
    }
    else
    {
        throw std::invalid_argument("Invalid input format: " + input_format_);
    }

    if (!dataset_.empty())
    {
        nncase::ptq_dataset_options ptq_options;
        ptq_options.dataset = dataset_;
        ptq_options.dataset_format = dataset_format_;
        ptq_options.calibrate_method = calibrate_method_;
        compiler->use_ptq(ptq_options);
    }

    compiler->compile();

    std::fstream out(output_filename_, std::ios::out | std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open output: " + output_filename_);
    compiler->gencode(out);
}
