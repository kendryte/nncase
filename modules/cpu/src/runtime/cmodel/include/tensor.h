#pragma once

#include <array>
#include <cstddef>
#include <gsl/gsl-lite.hpp>
#include <hardware_def.h>
#include <runtime_utils.h>

enum class tensor_loc_t : uint8_t {
    shared,
    device,
    local,
};

template <typename T, tensor_loc_t Loc = tensor_loc_t::local> class tensor {

  public:
    using value_type = T;

    tensor(dims_t dims)
        : dims_(dims),
          strides_(get_default_strides(dims_)),
          size_(compute_size(dims_)) {
        parent_ = nullptr;
        auto ptr = new T[size_];
        data_ = gsl::make_span(ptr, size_);
    }

    const dims_t &dimension() { return dims_; }
    const strides_t &strides() { return strides_; }

    gsl::span<T> data() { return data_; }

    gsl::span<const T> cdata() { return data_.template as_span<const T>(); }

    tensor<T, Loc> operator()(std::initializer_list<size_t> begins,
                              std::initializer_list<size_t> shapes) {
        return tensor(this, dims_t(begins.begin(), begins.end()),
                      dims_t(shapes.begin(), shapes.end()));
    }

    ~tensor() {
        if (parent_ == nullptr) {
            delete[] data_.data();
        }
    }

  private:
    tensor(tensor<T, Loc> *parent, dims_t begins, dims_t shapes)
        : parent_(parent),
          dims_(shapes.begin(), shapes.end()),
          strides_(parent->strides().begin(), parent->strides().end()) {
        size_ = compute_size(shapes);
        strides_ = parent->strides();
        data_ = parent->data_.subspan(offset(strides_, begins));
    }

    gsl::span<T> data_;
    void *parent_;
    dims_t dims_;
    strides_t strides_;
    std::size_t size_;
};