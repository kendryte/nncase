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
                         .add_argument(lyra::opt(input_type_, "input type").name("--input-type").optional().help("post trainning quantize input type, e.g float32|uint8|default, default is " + input_type_))
                         .add_argument(lyra::opt(output_type_, "output type").name("--output-type").optional().help("post trainning quantize output type, e.g float32|uint8, default is " + output_type_))
                         .add_argument(lyra::opt(quant_type_, "quant type").name("--quant-type").optional().help("post trainning quantize type, e.g uint8|int8, default is " + quant_type_))
                         .add_argument(lyra::opt(w_quant_type_, "pu output quant type").name("--pu-output-quant-type").optional().help("post trainning weights quantize type, e.g uint8|int8, default is " + w_quant_type_))
                         .add_argument(lyra::opt(input_layout_, "input layout").name("--input-layout").optional().help("input layout, e.g NCHW|NHWC, default is " + input_layout_))
                         .add_argument(lyra::opt(output_layout_, "output layout").name("--output-layout").optional().help("output layout, e.g nchw|default, default is " + output_layout_))
                         .add_argument(lyra::arg(input_filename_, "input file").required().help("input file"))
                         .add_argument(lyra::arg(output_filename_, "output file").required().help("output file"))
                         .add_argument(lyra::opt(input_prototxt_, "input prototxt").name("--input-prototxt").optional().help("input prototxt"))
                         .add_argument(lyra::opt(output_arrays_, "output arrays").name("--output-arrays").optional().help("output arrays"))
                         .add_argument(lyra::opt(dataset_, "dataset path").name("--dataset").optional().help("calibration dataset, used in post quantization"))
                         .add_argument(lyra::opt(dataset_format_, "dataset format").name("--dataset-format").optional().help("datset format: e.g. image, raw default is " + dataset_format_))
                         .add_argument(lyra::opt(calibrate_method_, "calibrate method").name("--calibrate-method").optional().help("calibrate method: e.g. no_clip, l2, default is " + calibrate_method_))
                         .add_argument(lyra::opt(input_mean_, "input mean").name("--input-mean").optional().help("input mean, default is " + std::to_string(input_mean_)))
                         .add_argument(lyra::opt(input_std_, "input std").name("--input-std").optional().help("input std, default is " + std::to_string(input_std_)))
                         .add_argument(lyra::opt(mean_, "normalize mean").name("--mean").optional().help("normalize mean, default is " + std::to_string(input_mean_)))
                         .add_argument(lyra::opt(scale_, "normalize scale").name("--scale").optional().help("normalize scale, default is " + std::to_string(input_std_)))
                         .add_argument(lyra::opt(image_format_, "image format").name("--image-format").optional().help("input image format, default is " + image_format_))
                         .add_argument(lyra::opt(input_range_, "input range").name("--input-range").optional())
                         .add_argument(lyra::opt(input_shape_, "input shape").name("--input-shape").optional())
                         .add_argument(lyra::opt(is_fpga_).name("--is-fpga").optional().help("use fpga parameters"))
                         .add_argument(lyra::opt(dump_ir_).name("--dump-ir").optional().help("dump ir to .dot"))
                         .add_argument(lyra::opt(dump_asm_).name("--dump-asm").optional().help("dump assembly"))
                         .add_argument(lyra::opt(dump_dir_, "dump directory").name("--dump-dir").optional().help("dump to directory"))
                         .add_argument(lyra::opt(benchmark_only_, "benchmark only").name("--benchmark-only").optional().help("compile kmodel only for benchmark use")));
}

void compile_command::run()
{
    if (!dataset_.empty())
    {
        if (input_type_ == "default")
            input_type_ = "uint8";
    }
    else
    {
        if (input_type_ == "default")
            input_type_ = "float32";
    }

    compile_options c_options;
    c_options.dump_asm = dump_asm_;
    c_options.dump_ir = dump_ir_;
    c_options.dump_dir = dump_dir_;
    c_options.target = target_name_;
    c_options.is_fpga = is_fpga_;
    c_options.input_type = input_type_;
    c_options.output_type = output_type_;
    c_options.quant_type = quant_type_;
    c_options.image_format = image_format_;
    c_options.mean = mean_;
    c_options.scale = scale_;
    c_options.input_range = input_range_;
    c_options.input_shape = input_shape_;
    c_options.w_quant_type = w_quant_type_;
    c_options.benchmark_only = benchmark_only_;

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
    i_options.input_layout = input_layout_;
    i_options.output_layout = output_layout_;

    auto compiler = nncase::compiler::create(c_options);
    if (input_format_ == "tflite")
    {
        auto file_data = read_file(input_filename_);
        compiler->import_tflite(file_data, i_options);
    }
    else if (input_format_ == "onnx")
    {
        auto file_data = read_file(input_filename_);
        compiler->import_onnx(file_data, i_options);
    }
    else if (input_format_ == "caffe")
    {
        if (input_prototxt_.empty())
            throw std::invalid_argument("Please use --input-prototxt to specify the path to the caffe prototxt");

        auto file_data = read_file(input_filename_);
        auto input_prototxt = read_file(input_prototxt_);
        compiler->import_caffe(file_data, input_prototxt);
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
        ptq_options.input_mean = input_mean_;
        ptq_options.input_std = input_std_;
        compiler->use_ptq(ptq_options);
    }

    compiler->compile();

    std::fstream out(output_filename_, std::ios::out | std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open output: " + output_filename_);
    compiler->gencode(out);
}
