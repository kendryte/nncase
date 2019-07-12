#include "nceval.h"
#include "dataset.h"
#include <clipp.h>
#include <filesystem>
#include <fstream>
#include <runtime/interpreter.h>

using namespace std;
using namespace clipp;
using namespace nncase::data;
using namespace nncase::runtime;

namespace
{
std::vector<uint8_t> read_file(const std::filesystem::path &filename)
{
    std::ifstream infile(filename, std::ios::binary | std::ios::in);
    if (infile.bad())
        throw std::runtime_error("Cannot open file: " + filename.string());

    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(length);
    infile.read(reinterpret_cast<char *>(data.data()), length);
    infile.close();
    return data;
}
}

struct eval_options
{
    std::filesystem::path model_filename;
    std::filesystem::path output_path;
    std::filesystem::path dataset;

    clipp::group parser()
    {
        return (
            value("model file", model_filename),
            value("output path", output_path),
            option("--dataset") & value("dataset path", dataset));
    }
};

void eval(const eval_options &options)
{
    auto model = read_file(options.model_filename);
    interpreter interp;
    if (!interp.try_load_model(model.data()))
        throw std::runtime_error("Invalid model");
}

int main(int argc, char *argv[])
{
    eval_options eval_options;
    auto cli = (eval_options.parser(),
        option("-v", "--version").call([] { cout << "version 1.0" << endl; }).doc("show version"));

    if (parse(argc, argv, cli))
    {
        try
        {
            eval(eval_options);
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

    return 0;
}
