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
#include <iostream>
#include <string>

using namespace std;
using namespace clipp;

int main(int argc, char *argv[])
{
    mode mode;
    compile_options compile_options;
    inference_options inference_options;

    auto cli = ((compile_options.parser(mode) | inference_options.parser(mode)),
        option("-v", "--version").call([] { cout << "version 0.2" << endl; }).doc("show version"));

    if (parse(argc, argv, cli))
    {
        try
        {
            switch (mode)
            {
            case mode::compile:
                compile(compile_options);
                break;
            case mode::inference:
                inference(inference_options);
                break;
            case mode::help:
                break;
            }

            return 0;
        }
        catch (std::exception &ex)
        {
            cerr << "Fatal: " << ex.what() << endl;
        }
    }
    else
    {
        auto fmt = doc_formatting {}.first_column(4).doc_column(28).last_column(80);
        cout << make_man_page(cli, "ncc", fmt).prepend_section("DESCRIPTION", "NNCASE model compiler and inference tool.") << endl;
    }

    return -1;
}
