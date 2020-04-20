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
#include "modes.h"
#include <iostream>
#include <string>

using namespace std;
using namespace clipp;

void help(clipp::group &cli, const char *program)
{
    auto fmt = doc_formatting {}.first_column(4).doc_column(28).last_column(80);
    cout << make_man_page(cli, program, fmt).prepend_section("DESCRIPTION", "NNCASE model compiler and inference tool.") << endl;
}

int main(int argc, char *argv[])
{
    mode mode;
    compile_options compile_options;
    inference_options inference_options;

    auto cli = ((compile_options.parser(mode) | inference_options.parser(mode) | option("-h", "--help").set(mode, mode::help).doc("show help")),
        option("-v", "--version").call([] { cout << "version 0.2" << endl; }).doc("show version"));

    if (argc == 1)
    {
        help(cli, "ncc");
        return 0;
    }

    auto parse_res = parse(argc, argv, cli);
    if (parse_res)
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
                help(cli, "ncc");
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
        for (const auto &m : parse_res.missing())
        {
            if (m.param()->flags().size())
                cout << "error: missing " << m.param()->flags()[0] << std::endl;
        }

        //per-argument mapping
        for (const auto &m : parse_res)
        {
            if (m.param()->flags().size() && m.any_error())
            {
                cout << "error: " << m.param()->flags()[0];

                if (m.blocked())
                    cout << " blocked" << std::endl;
                if (m.conflict())
                    cout << " conflict" << std::endl;
            }
        }

        cout << "Use -h to see help info" << std::endl;
    }

    return -1;
}
