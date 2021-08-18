// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_MAT_H
#define NCNN_MAT_H

#include "allocator.h"
#include <stdlib.h>
#include <string.h>

namespace ncnn
{

// the three dimension matrix
class Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w, size_t elemsize = 4u, Allocator *allocator = 0);
    // image
    Mat(int w, int h, size_t elemsize = 4u, Allocator *allocator = 0);
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4u, Allocator *allocator = 0);
    // packed vec
    Mat(int w, size_t elemsize, int elempack, Allocator *allocator = 0);
    // packed image
    Mat(int w, int h, size_t elemsize, int elempack, Allocator *allocator = 0);
    // packed dim
    Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator *allocator = 0);
    // copy
    Mat(const Mat &m);
    // external vec
    Mat(int w, void *data, size_t elemsize = 4u, Allocator *allocator = 0);
    // external image
    Mat(int w, int h, void *data, size_t elemsize = 4u, Allocator *allocator = 0);
    // external dim
    Mat(int w, int h, int c, void *data, size_t elemsize = 4u, Allocator *allocator = 0);
    // external packed vec
    Mat(int w, void *data, size_t elemsize, int elempack, Allocator *allocator = 0);
    // external packed image
    Mat(int w, int h, void *data, size_t elemsize, int elempack, Allocator *allocator = 0);
    // external packed dim
    Mat(int w, int h, int c, void *data, size_t elemsize, int elempack, Allocator *allocator = 0);
    // release
    ~Mat();
    // assign
    Mat &operator=(const Mat &m);
    // set all
    void fill(float v);
    // deep copy
    Mat clone(Allocator *allocator = 0) const;
    // deep copy from other mat, inplace
    void clone_from(const ncnn::Mat &mat, Allocator *allocator = 0);
    // reshape vec
    Mat reshape(int w, Allocator *allocator = 0) const;
    // reshape image
    Mat reshape(int w, int h, Allocator *allocator = 0) const;
    // reshape dim
    Mat reshape(int w, int h, int c, Allocator *allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator *allocator = 0);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator *allocator = 0);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator *allocator = 0);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, Allocator *allocator = 0);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator *allocator = 0);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator *allocator = 0);
    // allocate like
    void create_like(const Mat &m, Allocator *allocator = 0);
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // bits per element
    int elembits() const;

    // shape only
    Mat shape() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    float *row(int y);
    const float *row(int y) const;
    template <typename T>
    T *row(int y);
    template <typename T>
    const T *row(int y) const;

    // range reference
    Mat channel_range(int c, int channels);
    const Mat channel_range(int c, int channels) const;
    Mat row_range(int y, int rows);
    const Mat row_range(int y, int rows) const;
    Mat range(int x, int n);
    const Mat range(int x, int n) const;

    // access raw data
    template <typename T>
    operator T *();
    template <typename T>
    operator const T *() const;

    // convenient access float vec element
    float &operator[](size_t i);
    const float &operator[](size_t i) const;

    // convenient construct from half precision floating point data
    static Mat from_float16(const unsigned short *data, int size);

    // pointer to the data
    void *data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int *refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    Allocator *allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};

inline Mat::Mat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline Mat::Mat(int _w, size_t _elemsize, Allocator *_allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, Allocator *_allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, Allocator *_allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

inline Mat::Mat(int _w, size_t _elemsize, int _elempack, Allocator *_allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, int _elempack, Allocator *_allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator *_allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(const Mat &m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c), cstep(m.cstep)
{
    addref();
}

inline Mat::Mat(int _w, void *_data, size_t _elemsize, Allocator *_allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void *_data, size_t _elemsize, Allocator *_allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = (size_t)w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void *_data, size_t _elemsize, Allocator *_allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

inline Mat::Mat(int _w, void *_data, size_t _elemsize, int _elempack, Allocator *_allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void *_data, size_t _elemsize, int _elempack, Allocator *_allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = (size_t)w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void *_data, size_t _elemsize, int _elempack, Allocator *_allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

inline Mat::~Mat()
{
    release();
}

inline void Mat::fill(float _v)
{
    int size = (int)total();
    float *ptr = (float *)data;

    int remain = size;
    for (; remain > 0; remain--)
    {
        *ptr++ = _v;
    }
}

inline Mat &Mat::operator=(const Mat &m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline void Mat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Mat::total() const
{
    return cstep * c;
}

inline int Mat::elembits() const
{
    return elempack ? static_cast<int>(elemsize * 8) / elempack : 0;
}

inline Mat Mat::shape() const
{
    if (dims == 1)
        return Mat(w * elempack, (void *)0);
    if (dims == 2)
        return Mat(w, h * elempack, (void *)0);
    if (dims == 3)
        return Mat(w, h, c * elempack, (void *)0);

    return Mat();
}

inline Mat Mat::channel(int _c)
{
    return Mat(w, h, (unsigned char *)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::channel(int _c) const
{
    return Mat(w, h, (unsigned char *)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline float *Mat::row(int y)
{
    return (float *)((unsigned char *)data + (size_t)w * y * elemsize);
}

inline const float *Mat::row(int y) const
{
    return (const float *)((unsigned char *)data + (size_t)w * y * elemsize);
}

template <typename T>
inline T *Mat::row(int y)
{
    return (T *)((unsigned char *)data + (size_t)w * y * elemsize);
}

template <typename T>
inline const T *Mat::row(int y) const
{
    return (const T *)((unsigned char *)data + (size_t)w * y * elemsize);
}

inline Mat Mat::channel_range(int _c, int channels)
{
    return Mat(w, h, channels, (unsigned char *)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::channel_range(int _c, int channels) const
{
    return Mat(w, h, channels, (unsigned char *)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char *)data + (size_t)w * y * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char *)data + (size_t)w * y * elemsize, elemsize, elempack, allocator);
}

inline Mat Mat::range(int x, int n)
{
    return Mat(n, (unsigned char *)data + x * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::range(int x, int n) const
{
    return Mat(n, (unsigned char *)data + x * elemsize, elemsize, elempack, allocator);
}

template <typename T>
inline Mat::operator T *()
{
    return (T *)data;
}

template <typename T>
inline Mat::operator const T *() const
{
    return (const T *)data;
}

inline float &Mat::operator[](size_t i)
{
    return ((float *)data)[i];
}

inline const float &Mat::operator[](size_t i) const
{
    return ((const float *)data)[i];
}

} // namespace ncnn

#endif // NCNN_MAT_H
