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
#include "models/models.h"
#include <chrono>
#include <iostream>
#include <limits>
#include <nncase/runtime/interpreter.h>
#include <nncase/version.h>

using namespace nncase;
using namespace nncase::runtime;
namespace chrono = std::chrono;
using namespace std::chrono_literals;

size_t warm_up_count = 5;
size_t loop_count = 10;

result<void> bench_model(const std::string &name)
{
    interpreter interp;

    auto model = get_model(name);
    if (model.empty())
        return err(std::errc::no_such_file_or_directory);

    try_(interp.load_model(model));

    // warm up
    for (size_t i = 0; i < warm_up_count; i++)
    {
        try_(interp.run());
    }

    // run
    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max();
    double max_time = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < loop_count; i++)
    {
        auto start_time = chrono::steady_clock::now();
        try_(interp.run());
        auto end_time = chrono::steady_clock::now();
        auto duration_ns = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        auto duration_ms = duration_ns.count() / 1e6;

        total_time += duration_ms;
        min_time = std::min(min_time, duration_ms);
        max_time = std::max(max_time, duration_ms);
    }

    printf("%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", name.c_str(), min_time, max_time, total_time / loop_count);
    return ok();
}

const char *models[] = {
    "mnist",
    "mobilenet_v2"
};

int main()
{
    std::cout << "nncase Benchmark Tools " NNCASE_VERSION NNCASE_VERSION_SUFFIX << std::endl
              << "Copyright 2019-2021 Canaan Inc." << std::endl;

    for (size_t i = 0; i < sizeof(models) / sizeof(*models); i++)
    {
        auto r = bench_model(models[i]);
        if (r.is_err())
        {
            fprintf(stderr, "Cannot run %s: %s, skipped\n", models[i], r.unwrap_err().message().c_str());
        }
    }

    return 0;
}
