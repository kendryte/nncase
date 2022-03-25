/* Copyright 2019-2021 Canaan Inc.
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
#pragma once
#include <filesystem>
#include <functional>
#include <nncase/io_utils.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_tensor.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <xtensor/xarray.hpp>
#include <xtensor/xshape.hpp>

namespace nncase::data
{

struct data_batch
{
    std::vector<uint8_t> tensor;
    std::span<const std::filesystem::path> filenames;
};

class NNCASE_API dataset
{
public:
    class iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = data_batch;
        using pointer = data_batch *;
        using reference = data_batch &;

        bool operator==(const iterator &rhs) const noexcept { return from_ == rhs.from_; }
        bool operator!=(const iterator &rhs) const noexcept { return from_ != rhs.from_; }

        iterator &operator++()
        {
            value_ = dataset_->batch(from_ + dataset_->batch_size());
            if (value_)
                from_ += dataset_->batch_size();
            else
                *this = dataset_->end();
            return *this;
        }

        iterator &operator=(const iterator &rhs)
        {
            value_ = rhs.value_;
            from_ = rhs.from_;
            return *this;
        }

        data_batch &operator*()
        {
            if (value_)
                return *value_;

            throw std::runtime_error("Invalid datast iterator");
        }

        data_batch *operator->()
        {
            if (value_)
                return &value_.value();

            throw std::runtime_error("Invalid datast iterator");
        }

    private:
        friend class dataset;

        iterator(dataset &dataset, size_t from)
            : dataset_(&dataset), from_(from), value_(dataset.batch(from))
        {
        }

        dataset *dataset_;
        size_t from_;
        std::optional<data_batch> value_;
    };

    dataset(const std::filesystem::path &path, std::function<bool(const std::filesystem::path &)> file_filter, xt::dynamic_shape<size_t> input_shape, std::string input_layout);
    virtual ~dataset() = default;

    iterator begin()
    {
        return { *this, 0 };
    }

    iterator end()
    {
        return { *this, filenames_.size() };
    }

    size_t batch_size() const noexcept { return 1; }
    size_t total_size() const noexcept { return filenames_.size(); }

protected:
    virtual void process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) = 0;
    virtual void process(const std::vector<uint8_t> &src, uint8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) = 0;
    virtual void process(const std::vector<uint8_t> &src, int8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) = 0;
    virtual bool do_normalize() const noexcept { return true; }

private:
    std::optional<data_batch> batch(size_t from)
    {
        if (from + batch_size() <= filenames_.size())
        {
            size_t start = from;

            auto file = read_file(filenames_[from++]);
            // NOTE not support process
            // process(file, batch.data(), batch.shape(), input_layout_);
            std::span<const std::filesystem::path> filenames(filenames_.data() + start, filenames_.data() + from);

            return data_batch { file, filenames };
        }

        return {};
    }

private:
    std::vector<std::filesystem::path> filenames_;
    xt::dynamic_shape<size_t> input_shape_;
    std::string input_layout_;
};

class NNCASE_API image_dataset : public dataset
{
public:
    image_dataset(const std::filesystem::path &path, xt::dynamic_shape<size_t> input_shape, std::string input_layout);

protected:
    void process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) override;
    void process(const std::vector<uint8_t> &src, uint8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) override;
    void process(const std::vector<uint8_t> &src, int8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) override;
};

class NNCASE_API raw_dataset : public dataset
{
public:
    raw_dataset(const std::filesystem::path &path, xt::dynamic_shape<size_t> input_shape);

protected:
    void process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) override;
    void process(const std::vector<uint8_t> &src, uint8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) override;
    void process(const std::vector<uint8_t> &src, int8_t *dest, const xt::dynamic_shape<size_t> &shape, std::string layout) override;
    bool do_normalize() const noexcept override { return false; }
};
}
