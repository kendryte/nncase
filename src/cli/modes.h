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
#pragma once 
#include <string>
#include <clipp.h>

enum class mode
{
    compile,
    inference,
    help
};

struct compile_options
{
    std::string input_filename;
    std::string output_filename;
    std::string dataset;
    std::string input_format;
    std::string output_format = "kmodel";
    std::string target = "k210";
    std::string inference_type = "uint8";
    float input_mean = 0.f;
    float input_std = 1.f;
    bool dump_ir = false;

    clipp::group parser(mode &mode);
};

struct inference_options
{
    std::string model_filename;
    std::string output_path;
    std::string dataset;
    float input_mean = 0.f;
    float input_std = 1.f;

    clipp::group parser(mode &mode);
};

void compile(const compile_options &options);
void inference(const inference_options &options);
