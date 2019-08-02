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
    std::string input_format = "tflite";
    std::string output_format = "kmodel";
    std::string target = "k210";
    std::string inference_type = "uint8";
    float input_mean = 0.f;
    float input_std = 1.f;

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
