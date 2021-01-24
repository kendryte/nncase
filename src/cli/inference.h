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
#pragma once
#include <lyra/lyra.hpp>
#include <string>

namespace nncase::cli
{
class inference_command
{
public:
    inference_command(lyra::cli &cli);

private:
    void run();

private:
    std::string model_filename_;
    std::string output_path_;
    std::string dataset_;
    std::string dataset_format_ = "image";
    float input_mean_ = 0.f;
    float input_std_ = 1.f;
};
}
