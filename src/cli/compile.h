#pragma once 
#include <string>
#include <filesystem>
#include <clipp.h>

enum class mode
{
    compile,
    help
};

struct compile_options
{
    std::string input_filename;
    std::string output_filename;
    std::filesystem::path dataset;

    clipp::group parser(mode &mode);
};

void compile(const compile_options &compile_options);
