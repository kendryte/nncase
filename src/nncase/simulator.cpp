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
#include <nncase/data/dataset.h>
#include <nncase/io_utils.h>
#include <nncase/ir/debug.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/simulator.h>

using namespace nncase;
using namespace nncase::data;
using namespace nncase::runtime;

namespace
{
class simulator_impl : public simulator
{
public:
    simulator_impl(std::vector<uint8_t> model, const simulate_options &options)
        : model_(std::move(model)), options_(options)
    {
        interp_.load_model(gsl::as_bytes(gsl::make_span(model_))).unwrap_or_throw();
    }

    void run() override
    {
        if (!std::filesystem::exists(options_.output_path))
            std::filesystem::create_directories(options_.output_path);

        if (interp_.inputs_size() != 1)
            throw std::invalid_argument("Simulator only support models that have single 1 input");

        auto &in_shape = interp_.input_shape(0);
        xt::dynamic_shape<size_t> dataset_in_shape(in_shape.begin(), in_shape.end());
        std::unique_ptr<dataset> ds;
        if (options_.dataset_format == "image")
            ds = std::make_unique<image_dataset>(options_.dataset, dataset_in_shape, "NHWC", options_.input_mean, options_.input_std);
        else if (options_.dataset_format == "raw")
            ds = std::make_unique<raw_dataset>(options_.dataset, dataset_in_shape, options_.input_mean, options_.input_std);
        else
            throw std::runtime_error("Invalid dataset format: " + options_.dataset_format);

        auto in_type = interp_.input_desc(0).datatype;
        switch (in_type)
        {
        case dt_float32:
            eval<float>(*ds);
            break;
        case dt_uint8:
            eval<uint8_t>(*ds);
            break;
        case dt_int8:
            eval<int8_t>(*ds);
            break;
        default:
            throw std::runtime_error("Unsupported input datatype: " + std::string(datatype_names(in_type)));
        }
    }

private:
    template <class T>
    void eval(dataset &dataset)
    {
        size_t i = 0;
        for (auto it = dataset.begin<T>(); it != dataset.end<T>(); ++it)
        {
            auto input_buffer = host_runtime_tensor::buffer(interp_.input_tensor(0).unwrap()).unwrap_or_throw();
            auto &tensor = it->tensor;
            std::memcpy(input_buffer.data(), tensor.data(), input_buffer.size_bytes());

            auto r = interp_.run();
            if (r.is_ok())
            {
                std::filesystem::path out_filename(options_.output_path / it->filenames[0].filename());
                out_filename.replace_extension(".bin");

                std::ofstream of(out_filename, std::ios::binary | std::ios::out);
                for (size_t i = 0; i < interp_.outputs_size(); i++)
                {
                    auto output = host_runtime_tensor::buffer(interp_.output_tensor(i).unwrap()).unwrap_or_throw();
                    of.write(reinterpret_cast<const char *>(output.data()), output.size());
                }
            }
            else
            {
                std::cerr << "Eval " << it->filenames[0].filename() << " failed: " << r.unwrap_err().message() << std::endl;
            }

            if (options_.progress)
                options_.progress(i, dataset.total_size());
        }
    }

private:
    std::vector<uint8_t> model_;
    simulate_options options_;
    interpreter interp_;
};
}

simulator::~simulator()
{
}

std::unique_ptr<simulator> simulator::create(std::vector<uint8_t> model, const simulate_options &options)
{
    return std::make_unique<simulator_impl>(std::move(model), options);
}
