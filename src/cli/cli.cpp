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
#include "inference.h"
#include <nncase/version.h>

using namespace nncase::cli;

int main(int argc, char *argv[])
{
    std::cout << "nncase Command Line Tools " NNCASE_VERSION NNCASE_VERSION_SUFFIX << std::endl
              << "Copyright 2019-2021 Canaan Inc." << std::endl;

    bool show_version = false;
    auto cli = lyra::cli();
    cli.add_argument(lyra::opt(show_version).name("-v").name("--version").help("show version"));
    compile_command compile(cli);
    inference_command inference(cli);

    try
    {
        auto result = cli.parse({ argc, argv });
        if (argc == 1)
        {
            std::cout << cli;
            return 0;
        }

        if (!result)
        {
            std::cerr << result.errorMessage() << std::endl;
            return 1;
        }
    }
    catch (std::exception &ex)
    {
        std::cerr << "Fatal: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
