/* Copyright 2019-2020 Canaan Inc.
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
#include "inference.h"
#include "ProgressBar.hpp"
#include <nncase/io_utils.h>
#include <nncase/simulator.h>

using namespace nncase;
using namespace nncase::cli;

inference_command::inference_command(lyra::cli &cli)
{
    cli.add_argument(lyra::command("infer", [this](const lyra::group &) { this->run(); })
                         .add_argument(lyra::arg(model_filename_, "model filename").required().help("kmodel filename"))
                         .add_argument(lyra::arg(output_path_, "output path").required().help("output path"))
                         .add_argument(lyra::opt(dataset_, "dataset path").name("--dataset").required().help("dataset path"))
                         .add_argument(lyra::opt(dataset_format_, "dataset format").name("--dataset-format").optional().help("dataset format, e.g. image, raw, default is " + dataset_format_))
                         .add_argument(lyra::opt(input_mean_, "input mean").name("--input-mean").optional().help("input mean, default is " + std::to_string(input_mean_)))
                         .add_argument(lyra::opt(input_std_, "input std").name("--input-std").optional().help("input std, default is " + std::to_string(input_std_))));
}

void inference_command::run()
{
    simulate_options options;
    options.dataset = dataset_;
    options.dataset_format = dataset_format_;
    options.output_path = output_path_;
    options.input_mean = input_mean_;
    options.input_std = input_std_;

    auto sim = simulator::create(read_file(model_filename_), options);
    sim->run();
}
