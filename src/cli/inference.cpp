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
#include "modes.h"
#include <data/dataset.h>
#include <fstream>
#include <io_utils.h>
#include <kernels/k210/k210_kernels.h>
#include <runtime/runtime_op.h>
#include <runtime/target_interpreter.h>

using namespace clipp;
using namespace nncase;
using namespace nncase::data;
using namespace nncase::runtime;

namespace
{
struct eval_context
{
    interpreter_t interp;

    template <class T>
    void eval(const inference_options &options, dataset &dataset)
    {
        for (auto it = dataset.begin<T>(); it != dataset.end<T>(); ++it)
        {
            auto input = interp.input_at(0);
            auto input_mem = interp.memory_at<T>(input);
            auto &tensor = it->tensor;
            if (input.memory_type == mem_main)
            {
                std::copy(tensor.begin(), tensor.end(), input_mem.begin());
            }
            else if (input.memory_type == mem_k210_kpu)
            {
                if constexpr (std::is_same_v<T, uint8_t>)
                    kernels::k210::kpu_upload(tensor.data(), input_mem.data(), interp.input_shape_at(0));
                else
                    throw std::runtime_error("K210 only support uint8 input");
            }
            else
            {
                throw std::runtime_error("Unsupported input memory type");
            }

            interp.run(done_thunk, on_error_thunk, node_profile_thunk, this);

            boost::filesystem::path out_filename(options.output_path);

            out_filename /= it->filenames[0].filename();
            out_filename.replace_extension(".bin");

            std::ofstream of(out_filename.string(), std::ios::binary | std::ios::out);
            for (size_t i = 0; i < interp.outputs_size(); i++)
            {
                auto output = interp.memory_at<const char>(interp.output_at(i));
                of.write(output.data(), output.size());
            }
        }
    }

    void eval(const inference_options &options)
    {
        auto model = read_file(options.model_filename);
        if (!interp.try_load_model(model.data()))
            throw std::runtime_error("Invalid model");

        if (!boost::filesystem::exists(options.output_path))
            boost::filesystem::create_directories(options.output_path);

        auto in_shape = interp.input_shape_at(0);
        xt::dynamic_shape<size_t> shape { (size_t)in_shape[0], (size_t)in_shape[1], (size_t)in_shape[2], (size_t)in_shape[3] };
        std::unique_ptr<dataset> ds;
        if (options.dataset_format == "image")
            ds = std::make_unique<image_dataset>(options.dataset, shape, options.input_mean, options.input_std);
        else if (options.dataset_format == "raw")
            ds = std::make_unique<raw_dataset>(options.dataset, shape, options.input_mean, options.input_std);
        else
            throw std::runtime_error("Invalid dataset format: " + options.dataset_format);

        switch (interp.input_at(0).datatype)
        {
        case dt_float32:
            eval<float>(options, *ds);
            break;
        case dt_uint8:
            eval<uint8_t>(options, *ds);
            break;
        default:
            throw std::runtime_error("Unsupported input datatype");
        }

        std::cout << "Total: " << interp.total_duration().count() / 1e6 << "ms" << std::endl;
    }

    void on_done()
    {
    }

    static void done_thunk(void *userdata)
    {
        reinterpret_cast<eval_context *>(userdata)->on_done();
    }

    static void on_error_thunk(const char *err, void *userdata)
    {
        std::cerr << "Fatal: " << err << std::endl;
    }

    static void node_profile_thunk(runtime_opcode op, std::chrono::nanoseconds duration, void *userdata)
    {
        std::cout << node_opcode_names(op) << ": " << duration.count() / 1e6 << "ms" << std::endl;
    }
};
}

group inference_options::parser(mode &mode)
{
    return (
        command("infer").set(mode, mode::inference),
        "infer" % (value("input file", model_filename) % "input kmodel", value("output path", output_path) % "inference result output directory", required("--dataset") % "input dataset to inference" & value("dataset path", dataset), option("--dataset-format") % ("datset format: e.g. image, raw default is " + dataset_format) & value("dataset format", dataset_format), option("--input-mean") % ("input mean, default is " + std::to_string(input_mean)) & value("input mean", input_mean), option("--input-std") % ("input std, default is " + std::to_string(input_std)) & value("input std", input_std)));
}

void inference(const inference_options &options)
{
    eval_context ctx;
    ctx.eval(options);
}
