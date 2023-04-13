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
                         .add_argument(lyra::opt(input_format_, "input format").name("-i").name("--input-format").required().help("input format, e.g. tflite|onnx|caffe|pnnx"))
                         .add_argument(lyra::opt(target_name_, "target").name("-t").name("--target").required().help("target architecture, e.g. cpu|k210|k510"))
                         .add_argument(lyra::arg(input_filename_, "input file").required().help("input file"))
                         .add_argument(lyra::opt(input_prototxt_, "input prototxt").name("--input-prototxt").optional().help("input prototxt"))
                         .add_argument(lyra::arg(output_filename_, "output file").required().help("output file"))
                         .add_argument(lyra::opt(output_arrays_, "output arrays").name("--output-arrays").optional().help("output arrays"))
                         .add_argument(lyra::opt(quant_type_, "quant type").name("--quant-type").optional().help("post trainning quantize type, e.g uint8|int8|int16, default is " + quant_type_))
                         .add_argument(lyra::opt(w_quant_type_, "w quant type").name("--w-quant-type").optional().help("post trainning weights quantize type, e.g uint8|int8|int16, default is " + w_quant_type_))
                         .add_argument(lyra::opt(use_mse_quant_w_).name("--use-mse-quant-w").optional().help("use min mse algorithm to refine weights quantilization or not, default is " + std::to_string(use_mse_quant_w_)))
                         .add_argument(lyra::opt(split_w_to_act_).name("--split-w-to-act").optional().help("split weights to act or not, default is " + std::to_string(split_w_to_act_)))
                         .add_argument(lyra::opt(dataset_, "dataset path").name("--dataset").optional().help("calibration dataset, used in post quantization"))
                         .add_argument(lyra::opt(dataset_format_, "dataset format").name("--dataset-format").optional().help("datset format: e.g. image|raw, default is " + dataset_format_))
                         .add_argument(lyra::opt(dump_range_dataset_, "dataset path").name("--dump-range-dataset").optional().help("dump import op range dataset"))
                         .add_argument(lyra::opt(dump_range_dataset_format_, "dataset format").name("--dump-range-dataset-format").optional().help("datset format: e.g. image|raw, default is " + dump_range_dataset_format_))
                         .add_argument(lyra::opt(calibrate_method_, "calibrate method").name("--calibrate-method").optional().help("calibrate method: e.g. no_clip|l2|kld_m0|kld_m1|kld_m2|cdf, default is " + calibrate_method_))
                         .add_argument(lyra::opt(preprocess_).name("--preprocess").optional().help("enable preprocess, default is " + std::to_string(preprocess_)))
                         .add_argument(lyra::opt(swapRB_).name("--swapRB").optional().help("swap red and blue channel, default is " + std::to_string(swapRB_)))
                         .add_argument(lyra::opt(cli_mean_, "normalize mean").name("--mean").optional().help("normalize mean, default is " + cli_mean_))
                         .add_argument(lyra::opt(cli_std_, "normalize std").name("--std").optional().help("normalize std, default is " + cli_std_))
                         .add_argument(lyra::opt(cli_input_range_, "input range").name("--input-range").optional().help("float range after preprocess"))
                         .add_argument(lyra::opt(cli_output_range_, "output range").name("--output-range").optional().help("float range to quantize output"))
                         .add_argument(lyra::opt(cli_input_shape_, "input shape").name("--input-shape").optional().help("shape for input data"))
                         .add_argument(lyra::opt(letterbox_value_, "letter box value").name("--letterbox-value").optional().help("letter box pad value, default is " + std::to_string(letterbox_value_)))
                         .add_argument(lyra::opt(input_type_, "input type").name("--input-type").optional().help("input type, e.g float32|uint8|default, default is " + input_type_))
                         .add_argument(lyra::opt(output_type_, "output type").name("--output-type").optional().help("output type, e.g float32|uint8, default is " + output_type_))
                         .add_argument(lyra::opt(input_layout_, "input layout").name("--input-layout").optional().help("input layout, e.g NCHW|NHWC, default is " + input_layout_))
                         .add_argument(lyra::opt(output_layout_, "output layout").name("--output-layout").optional().help("output layout, e.g NCHW|NHWC, default is " + output_layout_))
                         .add_argument(lyra::opt(model_layout_, "model layout").name("--model-layout").optional().help("model layout, e.g NCHW|NHWC, default is empty"))
                         .add_argument(lyra::opt(is_fpga_).name("--is-fpga").optional().help("use fpga parameters, default is " + std::to_string(is_fpga_)))
                         .add_argument(lyra::opt(dump_ir_).name("--dump-ir").optional().help("dump ir to .dot, default is " + std::to_string(dump_ir_)))
                         .add_argument(lyra::opt(dump_asm_).name("--dump-asm").optional().help("dump assembly, default is " + std::to_string(dump_asm_)))
                         .add_argument(lyra::opt(dump_quant_error_).name("--dump-quant-error").optional().help("dump quant error, default is " + std::to_string(dump_quant_error_)))
                         .add_argument(lyra::opt(dump_import_op_range_).name("--dump-import-op-range").optional().help("dump import op range, default is " + std::to_string(dump_import_op_range_)))
                         .add_argument(lyra::opt(dump_dir_, "dump directory").name("--dump-dir").optional().help("dump to directory"))
                         .add_argument(lyra::opt(benchmark_only_).name("--benchmark-only").optional().help("compile kmodel only for benchmark use, default is " + std::to_string(benchmark_only_))));
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
    // manual parser the str to vector options
    mean_.clear(), std_.clear();
    parser_vector_opt(cli_mean_, mean_);
    parser_vector_opt(cli_std_, std_);
    parser_vector_opt(cli_input_range_, input_range_);
    parser_vector_opt(cli_input_shape_, input_shape_);
    parser_vector_opt(cli_output_range_, output_range_);

    compile_options c_options;
    c_options.dump_asm = dump_asm_;
    c_options.dump_ir = dump_ir_;
    c_options.dump_quant_error = dump_quant_error_;
    c_options.dump_import_op_range = dump_import_op_range_;
    c_options.dump_dir = dump_dir_;
    c_options.target = target_name_;
    c_options.is_fpga = is_fpga_;
    c_options.input_type = input_type_;
    c_options.output_type = output_type_;
    c_options.quant_type = quant_type_;
    c_options.swapRB = swapRB_;
    c_options.mean = mean_;
    c_options.std = std_;
    c_options.input_range = input_range_;
    c_options.output_range = output_range_;
    c_options.input_shape = input_shape_;
    c_options.w_quant_type = w_quant_type_;
    c_options.benchmark_only = benchmark_only_;
    c_options.preprocess = preprocess_;
    c_options.use_mse_quant_w = use_mse_quant_w_;
    c_options.split_w_to_act = split_w_to_act_;
    c_options.input_layout = input_layout_;
    c_options.output_layout = output_layout_;
    c_options.model_layout = model_layout_;
    c_options.letterbox_value = letterbox_value_;
    if (c_options.preprocess)
    {
        if (c_options.input_shape.empty())
        {
            throw std::invalid_argument("Empty input shape. If enable preprocess you must set input shape");
        }
        if (c_options.input_range.empty())
        {
            throw std::invalid_argument("Empty input range. If enable preprocess you must set input range");
        }
    }

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
    else if (input_format_ == "pnnx")
    {
        std::filesystem::path input_bin_filename_ = input_filename_;
        input_bin_filename_.replace_extension("bin");
        compiler->import_pnnx(input_filename_, input_bin_filename_.string(), i_options);
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

    if (!dump_range_dataset_.empty())
    {
        nncase::dump_range_dataset_options dump_range_options;
        dump_range_options.dataset = dump_range_dataset_;
        dump_range_options.dataset_format = dump_range_dataset_format_;
        dump_range_options.calibrate_method = calibrate_method_;
        compiler->dump_range_options(dump_range_options);
    }

    if (dump_import_op_range_ && dump_range_dataset_.empty())
        throw std::runtime_error("Dump range dataset has not been set.");

    compiler->compile();

    std::fstream out(output_filename_, std::ios::out | std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open output: " + output_filename_);
    compiler->gencode(out);
}
