#include "nceval.h"
#include "dataset.h"
#include <clipp.h>
#include <filesystem>
#include <fstream>
#include <runtime/interpreter.h>

using namespace std;
using namespace clipp;
using namespace nncase;
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

struct eval_context
{
    interpreter interp;

    template <class T>
    void eval(const eval_options &options, dataset &dataset)
    {
        for (auto it = dataset.begin<T>(); it != dataset.end<T>(); ++it)
        {
            auto input = interp.memory_at<T>(interp.input_at(0));
            auto &tensor = it->tensor;
            std::copy(tensor.begin(), tensor.end(), input.begin());

            interp.run(done_thunk, on_error_thunk, node_profile_thunk, this);

            std::filesystem::path out_filename(options.output_path);

            out_filename /= it->filenames[0].filename();
            out_filename.replace_extension(".bin");

            std::ofstream of(out_filename, std::ios::binary | std::ios::out);
            for (size_t i = 0; i < interp.outputs_size(); i++)
            {
                auto output = interp.memory_at<const char>(interp.output_at(i));
                of.write(output.data(), output.size());
            }
        }
    }

    void eval(const eval_options &options)
    {
        auto model = read_file(options.model_filename);
        if (!interp.try_load_model(model.data()))
            throw std::runtime_error("Invalid model");

        if (!std::filesystem::exists(options.output_path))
            std::filesystem::create_directories(options.output_path);

        auto in_shape = interp.input_shape_at(0);
        xt::dynamic_shape<size_t> shape { (size_t)in_shape[0], (size_t)in_shape[1], (size_t)in_shape[2], (size_t)in_shape[3] };
        image_dataset dataset(options.dataset, shape, 0.f, 1.f);

        switch (interp.input_at(0).datatype)
        {
        case dt_float32:
            eval<float>(options, dataset);
            break;
        case dt_uint8:
            eval<uint8_t>(options, dataset);
            break;
        default:
            throw std::runtime_error("Unsupported input datatype");
        }

        std::cout << "Total : " << interp.total_duration().count() / 1e6 << "ms" << std::endl;
    }

    void on_done()
    {
    }

    static void done_thunk(void *userdata)
    {
        reinterpret_cast<eval_context *>(userdata)->on_done();
    }

    static void on_error_thunk(const char *err, void *userdata)
    {
        std::cerr << "Fatal: " << err << std::endl;
    }

    static void node_profile_thunk(runtime_opcode op, std::chrono::nanoseconds duration, void *userdata)
    {
        std::cout << node_opcode_names(op) << ": " << duration.count() / 1e6 << "ms" << std::endl;
    }
};

int main(int argc, char *argv[])
{
    eval_options eval_options;
    auto cli = (eval_options.parser(),
        option("-v", "--version").call([] { cout << "version 1.0" << endl; }).doc("show version"));

    if (parse(argc, argv, cli))
    {
        try
        {
            eval_context ctx;
            ctx.eval(eval_options);
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
