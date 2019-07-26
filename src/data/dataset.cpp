#include <data/dataset.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace nncase;
using namespace nncase::data;

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

dataset::dataset(const std::filesystem::path &path, std::function<bool(const std::filesystem::path &)> file_filter, xt::dynamic_shape<size_t> input_shape, float mean, float std)
    : input_shape_(std::move(input_shape))
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

    if (filenames_.empty())
        throw std::invalid_argument("Invalid dataset, should contain one file at least");

    size_t samples = (filenames_.size() / batch_size()) * batch_size();
    filenames_.resize(samples);
}

std::optional<xt::xarray<float>> dataset::batch(size_t from)
{
    if (from + batch_size() <= filenames_.size())
    {
        xt::xarray<float> batch(input_shape_);
        for (auto &&sample_view : xt::split(batch, batch_size()))
        {
            auto view = xt::squeeze(sample_view, 0);
            auto file = read_file(filenames_[from++]);
            process(file, view.data(), view.shape());
        }

        return batch;
    }

    return {};
}

dataset::iterator dataset::begin()
{
    return { *this, 0 };
}

dataset::iterator dataset::end()
{
    return { *this, filenames_.size() };
}

dataset::iterator::iterator(dataset &dataset, size_t from)
    : dataset_(dataset), from_(from), value_(dataset.batch(from))
{
}

dataset::iterator &dataset::iterator::operator++()
{
    value_ = dataset_.batch(from_ + dataset_.batch_size());
    if (value_)
        from_ += dataset_.batch_size();
    else
        *this = dataset_.end();
    return *this;
}

dataset::iterator &dataset::iterator::operator=(const dataset::iterator &rhs)
{
    value_ = rhs.value_;
    from_ = rhs.from_;
    return *this;
}

xt::xarray<float> &dataset::iterator::operator*()
{
    if (value_)
        return *value_;

    throw std::runtime_error("Invalid datast iterator");
}

image_dataset::image_dataset(const std::filesystem::path &path, xt::dynamic_shape<size_t> input_shape, float mean, float std)
    : dataset(path, [](const std::filesystem::path &filename) { return cv::haveImageReader(filename.string()); }, std::move(input_shape), mean, std)
{
}

void image_dataset::process(const std::vector<uint8_t> &src, float *dest, const xt::dynamic_shape<size_t> &shape)
{
    auto img = cv::imdecode(src, cv::IMREAD_COLOR);

    cv::Mat f_img;
    if ((img.type() & CV_32F) == 0)
        img.convertTo(f_img, CV_32F, 1.0 / 255.0);
    else
        img.convertTo(f_img, CV_32F);

    cv::Mat dest_img;
    cv::resize(f_img, dest_img, cv::Size(shape[2], shape[1]));

    size_t channel_size = xt::compute_size(xt::dynamic_shape<size_t> { shape[1], shape[2] });
    dest_img.forEach<cv::Vec3f>([&](cv::Vec3f v, const int *idx) {
        auto i = idx[0] * shape[2] + idx[1];
        dest[i] = v[2];
        dest[i + channel_size] = v[1];
        dest[i + channel_size * 2] = v[0];
    });
}
