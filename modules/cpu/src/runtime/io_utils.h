#pragma once
#include <filesystem>
#include <fstream>
#include <gsl/gsl-lite.hpp>
#include <vector>

inline std::vector<uint8_t> read_stream(std::istream &stream) {
    stream.seekg(0, std::ios::end);
    size_t length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(length);
    stream.read(reinterpret_cast<char *>(data.data()), length);
    return data;
}

inline std::vector<uint8_t> read_file(const std::filesystem::path &filename) {
    std::ifstream infile(filename.string(), std::ios::binary | std::ios::in);
    if (!infile.good())
        throw std::runtime_error("Cannot open file: " + filename.string());
    return read_stream(infile);
}

template <class T>
inline void to_file(gsl::span<T> src, const std::filesystem::path &filename) {
    std::ofstream ofile(filename.string(), std::ios::binary | std::ios::out);
    ofile.write(src.template as_span<const char>().data(), src.size_bytes());
    ofile.close();
}