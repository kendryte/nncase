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
#include <fstream>
#include <magic_enum.hpp>
#include <nncase/data/dataset.h>
#include <nncase/io_utils.h>
#include <nncase/ir/debug.h>
#include <nncase/runtime/debug.h>
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

        auto dataset_paths = split(options_.dataset.string(), ":");
        if (interp_.inputs_size() != 1 && options_.dataset_format == "image")
            throw std::invalid_argument("Simulator only support models that have single 1 input for image dataset!");
        if (interp_.inputs_size() != dataset_paths.size())
            throw std::invalid_argument("Dataset Paths Must Equal To Inputs Number");

        std::vector<std::unique_ptr<dataset>> datasets;
        for (int i = 0; i < interp_.inputs_size(); i++)
        {
            auto &in_shape = interp_.input_shape(i);
            auto &path = dataset_paths[i];
            xt::dynamic_shape<size_t> dataset_in_shape(in_shape.begin(), in_shape.end());
            std::unique_ptr<dataset> ds;

            if (options_.dataset_format == "raw")
                ds = std::make_unique<raw_dataset>(std::filesystem::path(path), dataset_in_shape);
            else
                throw std::runtime_error("Invalid dataset format: " + options_.dataset_format);
            datasets.push_back(std::move(ds));
        }

        eval(std::move(datasets));
    }

private:
    std::vector<std::string> split(const std::string &str, const std::string &delim)
    {
        std::vector<std::string> tokens;
        size_t prev = 0, pos = 0;
        do
        {
            pos = str.find(delim, prev);
            if (pos == std::string::npos)
                pos = str.length();
            std::string token = str.substr(prev, pos - prev);
            if (!token.empty())
                tokens.push_back(token);
            prev = pos + delim.length();
        } while (pos < str.length() && prev < str.length());
        return tokens;
    }

    void eval(const std::vector<std::unique_ptr<dataset>> &datasets)
    {
        size_t count = 0;
        auto begins = std::vector<dataset::iterator>();
        auto ends = std::vector<dataset::iterator>();
        auto indexer = std::vector<int>(begins.size());
        std::iota(indexer.begin(), indexer.end(), 0);
        for (auto &ds : datasets)
        {
            begins.push_back(ds->begin());
            begins.push_back(ds->end());
        }
        while (std::accumulate(indexer.begin(), indexer.end(), true, [&](bool acc, int i)
            { return acc && begins[i] != ends[i]; }))
        {
            for (int j = 0; j < interp_.inputs_size(); j++)
            {
                auto input_tensor = interp_.input_tensor(j).unwrap();
                auto input_map = std::move(hrt::map(input_tensor, hrt::map_write).unwrap());
                auto input_buffer = input_map.buffer();
                auto &tensor = begins[j]->tensor;
                std::memcpy(input_buffer.data(), tensor.data(), input_buffer.size_bytes());
            }

            auto r = interp_.run();
            if (r.is_ok())
            {
                std::filesystem::path out_filename(options_.output_path / ("out" + std::to_string(count)));
                out_filename.replace_extension(".bin");

                std::ofstream of(out_filename, std::ios::binary | std::ios::out);
                for (size_t i = 0; i < interp_.outputs_size(); i++)
                {
                    auto output_tensor = interp_.output_tensor(i).unwrap();
                    auto output_map = std::move(hrt::map(output_tensor, hrt::map_read).unwrap());
                    auto output_buffer = output_map.buffer();
                    of.write(reinterpret_cast<const char *>(output_buffer.data()), output_buffer.size());
                }
            }
            else
            {
                std::cerr << "Eval " << std::to_string(count) << " failed: " << r.unwrap_err().message() << std::endl;
            }

            // todo current not support
            // if (options_.progress)
            //     options_.progress(i, dataset.total_size());

            for (auto &it : begins)
            {
                ++it;
            }
            count++;
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
