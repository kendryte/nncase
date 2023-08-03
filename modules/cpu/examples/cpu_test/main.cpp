#include <chrono>
#include <fstream>
#include <iostream>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

template <class T>
std::vector<T> read_binary_file(const char *file_name)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    std::vector<T> vec(len / sizeof(T), 0);
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(vec.data()), len);
    ifs.close();
    return vec;
}

void read_binary_file(const char *file_name, char *buffer)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.read(buffer, len);
    ifs.close();
}

auto read_binary(const char *file_name, char *buffer, size_t begin, size_t count)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(begin, ifs.beg);
    ifs.read(buffer + begin, count);
    ifs.close();
    std::cout << "read bin seg ok" << std::endl;
}

size_t get_binary_file_size(const char *file_name)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    ifs.close();
    return len;
}

auto load_bin_to_kmodel(char *file_path, interpreter &interp)
{
    auto model_size = get_binary_file_size(file_path);
    auto model_data = std::make_unique<char[]>(model_size);
    for (size_t i = 0; i < model_size;)
    {
        size_t count = 8000000;
        if (count + i >= model_size)
            count = model_size - i;
        read_binary(file_path, model_data.get(), i, count);
        i += 8000000;
    }
    // show the identification of kmodel.
    // print_one_line_data("kmodel identification :", model_data.get(), 4);

    interp.load_model({ (const gsl::byte *)model_data.get(), model_size }).expect("cannot load kmodel.");
    std::cout << "load kmodel success" << std::endl;

    return model_data;
}

template <typename T>
double dot(const T *v1, const T *v2, size_t size)
{
    double ret = 0.f;
    for (size_t i = 0; i < size; i++)
    {
        ret += v1[i] * v2[i];
    }

    return ret;
}

template <typename T>
double cosine(const T *v1, const T *v2, size_t size)
{
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << v1[i] << " " << v2[i] << std::endl;
    }
    return dot(v1, v2, size) / ((sqrt(dot(v1, v1, size)) * sqrt(dot(v2, v2, size))));
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;

    if (argc < 3)
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <kmodel> <input_0.bin> <input_1.bin> ... <input_N.bin> " << std::endl;
        std::cerr << argv[0] << " <kmodel> <input_0.bin> <input_1.bin> ... <input_N.bin> <output_0.bin> <output_1.bin> ... <output_N.bin>" << std::endl;
        return -1;
    }

    interpreter interp;

    // 1. load model
    std::ifstream ifs(argv[1], std::ios::binary);
    interp.load_model(ifs).expect("Invalid kmodel");

    // 2. set inputs
    for (size_t i = 2, j = 0; i < 2 + interp.inputs_size(); i++, j++)
    {
        auto desc = interp.input_desc(j);
        auto shape = interp.input_shape(j);
        auto tensor = hrt::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create input tensor");
        interp.input_tensor(j, tensor).expect("cannot set input tensor");

        auto span = tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
        read_binary_file(argv[i], reinterpret_cast<char *>(span.data()));
        hrt::sync(tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
    }

    // 3. set outputs
    // for (size_t i = 0; i < interp.outputs_size(); i++)
    // {
    //     auto desc = interp.output_desc(i);
    //     auto shape = interp.output_shape(i);
    //     auto tensor = hrt::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create input tensor");
    //     interp.output_tensor(i, tensor).expect("cannot set output tensor");
    // }

    // 4. run
    interp.run().expect("error occurred in running model");
    auto start = std::chrono::steady_clock::now();
    interp.run().expect("error occurred in running model");
    auto stop = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(stop - start).count();

    // 5. get outputs
    double cos = 0.f;
    for (int i = 2 + interp.inputs_size(), j = 0; i < argc; i++, j++)
    {
        auto desc = interp.output_desc(j);
        auto out = interp.output_tensor(j).expect("cannot get output tensor");
        auto mapped_buf = std::move(hrt::map(out, map_access_t::map_read).unwrap());
        auto vec = read_binary_file<unsigned char>(argv[i]);
        switch (desc.datatype)
        {
        case dt_boolean:
        case dt_uint8:
        {
            cos = cosine((const uint8_t *)mapped_buf.buffer().data(), (const uint8_t *)vec.data(), vec.size() / sizeof(uint8_t));
            break;
        }
        case dt_int8:
        {
            cos = cosine((const int8_t *)mapped_buf.buffer().data(), (const int8_t *)vec.data(), vec.size() / sizeof(int8_t));
            break;
        }
        case dt_float32:
        {
            cos = cosine((const float *)mapped_buf.buffer().data(), (const float *)vec.data(), vec.size() / sizeof(float));
            break;
        }
        case dt_int32:
        {
            cos = cosine((const int32_t *)mapped_buf.buffer().data(), (const int32_t *)vec.data(), vec.size() / sizeof(int32_t));
            break;
        }
        case dt_int64:
        {
            cos = cosine((const int64_t *)mapped_buf.buffer().data(), (const int64_t *)vec.data(), vec.size() / sizeof(int64_t));
            break;
        }
        default:
        {
            std::cerr << "not supported data type: " << desc.datatype << std::endl;
            std::abort();
        }
        }

        std::cout << "output " << j << " cosine similarity: " << cos << std::endl;
    }

    std::cout << "interp run: " << duration << " ms, fps = " << 1000 / duration << std::endl;

    return 0;
}