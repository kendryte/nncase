#pragma once 
#include <string>
#include <filesystem>
#include <clipp.h>

enum class mode
{
    compile,
    inference,
    help
};

struct compile_options
{
    std::filesystem::path input_filename;
    std::filesystem::path output_filename;
    std::filesystem::path dataset;
    std::string inference_type = "uint8";

    clipp::group parser(mode &mode);
};

struct inference_options
{
    std::filesystem::path model_filename;
    std::filesystem::path output_path;
    std::filesystem::path dataset;

    clipp::group parser(mode &mode);
};

void compile(const compile_options &options);
void inference(const inference_options &options);
