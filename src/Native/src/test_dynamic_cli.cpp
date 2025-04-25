#include <chrono>
#include <fstream>
#include <iostream>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <string>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

template <class T>
void print_one_line_data(std::string name, std::vector<T> tmp,
                         size_t count = 0) {
    std::cout << name << " data, size = " << tmp.size() << std::endl;
    if (count == 0)
        count = tmp.size();
    std::cout << "[ ";
    for (size_t i = 0; i < count; i++)
        std::cout << (int64_t)tmp[i] << " ";
    std::cout << " ]" << std::endl;
}

template <class T>
void print_one_line_data(std::string name, T tmp, size_t count = 0) {
    std::cout << name << " data " << std::endl;
    std::cout << "[ ";
    for (size_t i = 0; i < count; i++)
        std::cout << (char)tmp[i] << " ";
    std::cout << " ]" << std::endl;
}

template <class T>
std::vector<std::vector<T>> txt_to_vector(const char *pathname) {
    std::ifstream infile;
    infile.open(pathname);

    std::vector<std::vector<T>> res;
    std::vector<T> suanz;
    std::string s;

    while (getline(infile, s)) {
        std::istringstream is(s);
        T d;
        while (!is.eof()) {
            is >> d;
            suanz.push_back(d);
        }
        res.push_back(suanz);

        suanz.clear();
        s.clear();
    }

    infile.close();

    return res;
}

std::vector<unsigned char> read_binary_file(const char *file_name) {
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    std::vector<unsigned char> vec(len / sizeof(unsigned char), 0);
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(vec.data()), len);
    ifs.close();
    return vec;
}

size_t read_binary_file(const char *file_name, char *buffer) {
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.read(buffer, len);
    ifs.close();
    return len;
}

auto read_binary(const char *file_name, char *buffer, size_t begin,
                 size_t count) {
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(begin, ifs.beg);
    ifs.read(buffer + begin, count);
    ifs.close();
}

size_t get_binary_file_size(const char *file_name) {
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    ifs.close();
    return len;
}

template <typename T> double dot(const T *v1, const T *v2, size_t size) {
    double ret = 0.f;
    for (size_t i = 0; i < size; i++) {
        ret += v1[i] * v2[i];
    }

    return ret;
}

template <typename T> double cosine(const T *v1, const T *v2, size_t size) {
    return dot(v1, v2, size) /
           ((sqrt(dot(v1, v1, size)) * sqrt(dot(v2, v2, size))));
}

result<std::vector<value_t>> to_values(value_t v) {
    if (v.is_a<tensor>()) {
        return ok(std::vector{v});
    } else if (v.is_a<tuple>()) {
        auto out_fields = v.as<tuple>().unwrap()->fields();
        return ok(std::vector(out_fields.begin(), out_fields.end()));
    } else {
        return err(std::errc::invalid_argument);
    }
}

// template <class T>
// auto get_result(std::vector<nncase::object_t<nncase::value_node>> &values,
// int out_idx)
// {
//     auto t = values[out_idx].as<tensor>().expect("value is not a tensor");
//     auto d = (T *)get_output_span(t).unwrap().data();
//     std::vector<T> dec_result(d, d + t->length());
//     auto dec_shape = t->shape();
//     return std::make_tuple(dec_result, dec_shape);
// }

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

static dims_t parse_digits(const std::string &s) {
    dims_t digits;
    auto strs = split(s, ' ');
    for (size_t i = 0; i < strs.size(); i++) {
        digits.push_back(std::stoi(strs[i]));
    }
    return digits;
}

static std::vector<dims_t>
parse_multilines(const std::vector<std::string> &strs, size_t begin,
                 size_t size) {
    std::vector<dims_t> result;
    for (auto i = strs.begin() + begin; i != strs.begin() + begin + size; ++i) {
        auto shape = parse_digits(*i);
        if (shape[0] == 0) {
            shape = dims_t{};
        }
        result.push_back(shape);
    }
    return result;
}

struct data_desc {
    std::vector<dims_t> input_shape;
    std::vector<dims_t> output_shape;
    bool is_empty() { return input_shape.empty() && output_shape.empty(); }
};

data_desc parse_desc(const unsigned char *kmodel_desc_raw) {
    auto kmode_desc =
        std::string(reinterpret_cast<const char *>(kmodel_desc_raw));
    auto descs = split(kmode_desc, '\n');
    auto nums = parse_digits(descs[0]);
    auto input_num = nums[0];
    auto output_num = nums[1];
    auto in_shapes = parse_multilines(descs, 1, input_num);
    auto out_shapes = parse_multilines(descs, 1 + input_num, output_num);
    return data_desc{in_shapes, out_shapes};
}

template <class T> float compare_output(tensor t, std::vector<T> expect) {
    auto unmap_buf = t->to_host()
                         .expect("not host")
                         ->buffer()
                         .as_host()
                         .expect("not host buffer");
    auto mapped_buf = std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    int ret = memcmp((void *)mapped_buf.buffer().data(), (void *)expect.data(),
                     expect.size() * sizeof(T));
    if (!ret) {
        return 1;
    } else {
        float cos =
            cosine((float *)mapped_buf.buffer().data(), (float *)expect.data(),
                   expect.size() * sizeof(T) / sizeof(float));
        return cos;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__
              << std::endl;
    if (argc < 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " <kmodel> <.desc> <input_0.bin> <input_1.bin> ... "
               "<input_N.bin> <output_0.bin> <output_1.bin> ... <output_N.bin>"
            << std::endl;
        return -1;
    }
    interpreter interp;
    std::ifstream ifs(argv[1], std::ios::binary);
    nncase::runtime::std_istream stream(ifs);
    interp.load_model(stream).expect("Invalid kmodel");

    auto entry = interp.entry_function().expect("entry function is nullptr");

    auto input_data_desc = parse_desc(read_binary_file(argv[2]).data());
    std::vector<value_t> inputs;
    for (size_t i = 0; i < interp.inputs_size(); i++) {
        auto type =
            entry->parameter_type(i).expect("parameter type out of index");
        auto ts_type =
            type.as<tensor_type>().expect("input is not a tensor type");

        auto in_data = read_binary_file(argv[i + 3]);
        auto data_type = ts_type->dtype()->typecode();

        auto tensor = host_runtime_tensor::create(
                          data_type, input_data_desc.input_shape[i],
                          {(std::byte *)in_data.data(), (size_t)in_data.size()},
                          true, hrt::pool_shared)
                          .expect("cannot create input tensor");
        hrt::sync(tensor, sync_op_t::sync_write_back, true).unwrap();
        inputs.push_back(tensor.impl());
    }

    auto start = std::chrono::steady_clock::now();
    auto return_value = interp.entry_function()
                            .expect("no entry_function")
                            ->invoke(inputs)
                            .expect("run entry_function failed");
    auto stop = std::chrono::steady_clock::now();
    double duration =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "interp run: " << duration << " ms, fps = " << 1000 / duration
              << std::endl;
    auto values = to_values(return_value).expect("unsupported value type");

    for (size_t i = 0; i < values.size(); ++i) {
        auto t = values[i].as<tensor>().expect("value is not a tensor");
        auto cos = compare_output(
            t, read_binary_file(argv[i + 3 + interp.inputs_size()]));
        std::cout << "compare output [" << i << "] cosine similarity = " << cos
                  << std::endl;
    }

    std::cout << "}" << std::endl;

    return 0;
}
