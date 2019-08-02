// nncase-cli.cpp: 定义应用程序的入口点。
//

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
        cout << usage_lines(cli, "ncc") << endl;
    }

    return -1;
}
