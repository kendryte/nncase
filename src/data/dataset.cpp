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
#include <fstream>
#include <nncase/data/dataset.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace nncase;
using namespace nncase::data;

dataset::dataset(const std::filesystem::path &path, std::function<bool(const std::filesystem::path &)> file_filter, xt::dynamic_shape<size_t> input_shape, std::string input_layout, float mean, float std)
    : input_shape_(std::move(input_shape)), input_layout_(input_layout), mean_(mean), std_(std)
{
    if (std::filesystem::is_directory(path))
    {
        for (auto &&filename : std::filesystem::recursive_directory_iterator(path))
        {
            if (file_filter(filename))
                filenames_.emplace_back(filename);
        }
    }
    else if (std::filesystem::exists(path))
    {
        if (file_filter(path))
            filenames_.emplace_back(path);
    }

    size_t samples = (filenames_.size() / batch_size()) * batch_size();
    filenames_.resize(samples);

    if (filenames_.empty())
        throw std::invalid_argument("Invalid dataset, should contain one file at least");
}

image_dataset::image_dataset(const std::filesystem::path &path, xt::dynamic_shape<size_t> input_shape, std::string input_layout, float mean, float std)
    : dataset(
        path, [](const std::filesystem::path &filename) { return cv::haveImageReader(filename.string()); },
        std::move(input_shape), input_layout, mean, std)
{
}

void image_dataset::process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape, std::string layout)
{
    auto img = cv::imdecode(src, cv::IMREAD_COLOR);

    cv::Mat f_img;
    if ((img.type() & CV_32F) == 0)
        img.convertTo(f_img, CV_32F, 1.0 / 255.0);
    else
        img.convertTo(f_img, CV_32F);

    cv::Mat dest_img;
    if (layout == "NHWC")
    {
        cv::resize(f_img, dest_img, cv::Size((int)shape[2], (int)shape[1]));

        if (shape[3] == 3)
        {
            dest_img.forEach<cv::Vec3f>([&](cv::Vec3f v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i * 3] = v[2];
                dest[i * 3 + 1] = v[1];
                dest[i * 3 + 2] = v[0];
            });
        }
        else if (shape[3] == 1)
        {
            dest_img.forEach<cv::Vec3f>([&](cv::Vec3f v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[0];
            });
        }
        else
        {
            throw std::runtime_error("Unsupported image channels: " + std::to_string(shape[0]));
        }
    }
    else if (layout == "NCHW")
    {
        cv::resize(f_img, dest_img, cv::Size((int)shape[3], (int)shape[2]));

        size_t channel_size = xt::compute_size(xt::dynamic_shape<size_t> { shape[2], shape[3] });
        if (shape[1] == 3)
        {
            dest_img.forEach<cv::Vec3f>([&](cv::Vec3f v, const int *idx) {
                auto i = idx[0] * shape[3] + idx[1];
                dest[i] = v[2];
                dest[i + channel_size] = v[1];
                dest[i + channel_size * 2] = v[0];
            });
        }
        else if (shape[1] == 1)
        {
            dest_img.forEach<cv::Vec3f>([&](cv::Vec3f v, const int *idx) {
                auto i = idx[0] * shape[3] + idx[1];
                dest[i] = v[0];
            });
        }
        else
        {
            throw std::runtime_error("Unsupported image channels: " + std::to_string(shape[0]));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported layout type!");
    }
}

void image_dataset::process(const std::vector<uint8_t> &src, uint8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout)
{
    auto img = cv::imdecode(src, cv::IMREAD_COLOR);

    cv::Mat f_img;
    if ((img.type() & CV_8U) == 0)
        img.convertTo(f_img, CV_8U);
    else
        img.convertTo(f_img, CV_8U);

    cv::Mat dest_img;
    if (layout == "NHWC")
    {
        cv::resize(f_img, dest_img, cv::Size((int)shape[2], (int)shape[1]));

        if (shape[0] == 3)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i * 3] = v[2];
                dest[i * 3 + 1] = v[1];
                dest[i * 3 + 2] = v[0];
            });
        }
        else if (shape[0] == 1)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[0];
            });
        }
        else
        {
            throw std::runtime_error("Unsupported image channels: " + std::to_string(shape[0]));
        }
    }
    else if (layout == "NCHW")
    {
        cv::resize(f_img, dest_img, cv::Size((int)shape[3], (int)shape[2]));

        size_t channel_size = xt::compute_size(xt::dynamic_shape<size_t> { shape[2], shape[3] });
        if (shape[1] == 3)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[2];
                dest[i + channel_size] = v[1];
                dest[i + channel_size * 2] = v[0];
            });
        }
        else if (shape[1] == 1)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[0];
            });
        }
        else
        {
            throw std::runtime_error("Unsupported image channels: " + std::to_string(shape[0]));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported layout type!");
    }
}

void image_dataset::process(const std::vector<uint8_t> &src, int8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout)
{
    auto img = cv::imdecode(src, cv::IMREAD_COLOR);

    cv::Mat f_img;
    if ((img.type() & CV_8S) == 0)
        img.convertTo(f_img, CV_8S);
    else
        img.convertTo(f_img, CV_8S);

    cv::Mat dest_img;
    if (layout == "NHWC")
    {
        cv::resize(f_img, dest_img, cv::Size((int)shape[2], (int)shape[1]));

        if (shape[0] == 3)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i * 3] = v[2];
                dest[i * 3 + 1] = v[1];
                dest[i * 3 + 2] = v[0];
            });
        }
        else if (shape[0] == 1)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[0];
            });
        }
        else
        {
            throw std::runtime_error("Unsupported image channels: " + std::to_string(shape[0]));
        }
    }
    else if (layout == "NCHW")
    {
        cv::resize(f_img, dest_img, cv::Size((int)shape[3], (int)shape[2]));

        size_t channel_size = xt::compute_size(xt::dynamic_shape<size_t> { shape[2], shape[3] });
        if (shape[1] == 3)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[2];
                dest[i + channel_size] = v[1];
                dest[i + channel_size * 2] = v[0];
            });
        }
        else if (shape[1] == 1)
        {
            dest_img.forEach<cv::Vec3b>([&](cv::Vec3b v, const int *idx) {
                auto i = idx[0] * shape[2] + idx[1];
                dest[i] = v[0];
            });
        }
        else
        {
            throw std::runtime_error("Unsupported image channels: " + std::to_string(shape[0]));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported layout type!");
    }
}

raw_dataset::raw_dataset(const std::filesystem::path &path, xt::dynamic_shape<size_t> input_shape, float mean, float std)
    : dataset(
        path, []([[maybe_unused]] const std::filesystem::path &filename) { return true; },
        std::move(input_shape), "", mean, std)
{
}

void raw_dataset::process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape, [[maybe_unused]] std::string layout)
{
    auto expected_size = xt::compute_size(shape) * sizeof(float);
    auto actual_size = src.size();
    if (expected_size != actual_size)
    {
        throw std::runtime_error("Invalid dataset, file size should be "
            + std::to_string(expected_size) + "B, but got " + std::to_string(actual_size) + "B");
    }

    auto data = reinterpret_cast<const float *>(src.data());
    std::copy(data, data + actual_size / sizeof(float), dest);
}

void raw_dataset::process(const std::vector<uint8_t> &src, uint8_t *dest, const xt::dynamic_shape<size_t> &shape, [[maybe_unused]] std::string layout)
{
    auto expected_size = xt::compute_size(shape);
    auto actual_size = src.size();
    if (expected_size != actual_size)
    {
        throw std::runtime_error("Invalid dataset, file size should be "
            + std::to_string(expected_size) + "B, but got " + std::to_string(actual_size) + "B");
    }

    std::copy(src.begin(), src.end(), dest);
}

void raw_dataset::process(const std::vector<uint8_t> &src, int8_t *dest, const xt::dynamic_shape<size_t> &shape, [[maybe_unused]] std::string layout)
{
    auto expected_size = xt::compute_size(shape);
    auto actual_size = src.size();
    if (expected_size != actual_size)
    {
        throw std::runtime_error("Invalid dataset, file size should be "
            + std::to_string(expected_size) + "B, but got " + std::to_string(actual_size) + "B");
    }

    auto data = reinterpret_cast<const int8_t *>(src.data());
    std::copy(data, data + actual_size / sizeof(int8_t), dest);
}
