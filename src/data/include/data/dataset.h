#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <optional>
#include <xtensor/xarray.hpp>
#include <xtensor/xshape.hpp>

namespace nncase
{
namespace data
{
    class dataset
    {
    public:
        class iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = xt::xarray<float>;
            using pointer = xt::xarray<float> *;
            using reference = xt::xarray<float> &;

            bool operator==(const iterator &rhs) const noexcept { return from_ == rhs.from_; }
            bool operator!=(const iterator &rhs) const noexcept { return from_ != rhs.from_; }
            iterator &operator++();
            iterator &operator=(const iterator &rhs);

            xt::xarray<float> &operator*();

        private:
            friend class dataset;

            iterator(dataset &dataset, size_t from);

            dataset &dataset_;
            size_t from_;
            std::optional<xt::xarray<float>> value_;
        };

        dataset(const std::filesystem::path &path, std::function<bool(const std::filesystem::path &)> file_filter, xt::dynamic_shape<size_t> input_shape, float mean, float std);

        iterator begin();
        iterator end();

        size_t batch_size() const noexcept { return input_shape_[0]; }

    protected:
        virtual void process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape) = 0;

    private:
        std::optional<xt::xarray<float>> batch(size_t from);

    private:
        std::vector<std::filesystem::path> filenames_;
        xt::dynamic_shape<size_t> input_shape_;
    };

    class image_dataset : public dataset
    {
    public:
        image_dataset(const std::filesystem::path &path, xt::dynamic_shape<size_t> input_shape, float mean, float std);

    protected:
        void process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape) override;
    };
}
}
