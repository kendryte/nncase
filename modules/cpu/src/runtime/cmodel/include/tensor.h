#pragma once

#include <cstddef>
#include "../../gsl-lite.hpp"
#include <hardware_def.h>
#include <runtime_utils.h>
#include "../../method_table_def.h"

enum class loc_t : uint8_t {
    shared,
    device,
    local,
};

template <typename T, loc_t Loc = loc_t::local> class tensor {

  public:
    using value_type = T;
    tensor(tensor<T, Loc> &) = delete;
    tensor &operator=(tensor<T, Loc> &) = delete;

    tensor(dims_t dims)
        : dims_(dims),
          strides_(get_default_strides(dims_)),
          size_(compute_size(dims_)) {
        parent_ = nullptr;
        auto ptr = (T *)runtime_util->malloc(size_ * sizeof(T));
        data_ = gsl::make_span(ptr, size_);
    }

    tensor(gsl::span<T> data, dims_t dims)
        : data_(data),
          parent_(data.data()),
          dims_(dims),
          strides_(get_default_strides(dims_)),
          size_(compute_size(dims_)) {
        if (size_ != data_.size()) {
            runtime_util->rt_assert(false, (char*)"Invalid tensor size");
        }
    }

    tensor(gsl::span<T> data, dims_t dims, strides_t strides)
        : data_(data),
          parent_(data.data()),
          dims_(dims),
          strides_(strides),
          size_(compute_size(dims_, strides_)) {
        if (size_ != data_.size()) {
            runtime_util->rt_assert(false, (char*)"Invalid tensor size");
        }
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

    tensor<T, Loc> operator()(dims_t begins, dims_t shapes) {
        return tensor(this, begins, shapes);
    }

    ~tensor() {
        if (parent_ == nullptr) {
            runtime_util->free(data_.data());
        }
    }

  private:
    tensor(tensor<T, Loc> *parent, dims_t begins, dims_t shapes)
        : parent_(parent),
          dims_(shapes.begin(), shapes.end()),
          strides_(parent->strides().begin(), parent->strides().end()) {
        size_ = compute_size(shapes);
        strides_ = parent->strides();
        auto subspan_offset = offset(strides_, begins);
        data_ = parent->data_.subspan(subspan_offset);
        if (data_.size() < size_) {
            runtime_util->rt_assert(false, (char*)"Invalid tensor size");
        }
    }

    gsl::span<T> data_;
    void *parent_;
    dims_t dims_;
    strides_t strides_;
    size_t size_;
};