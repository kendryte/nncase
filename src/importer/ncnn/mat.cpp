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

#include "mat.h"

#include <math.h>

namespace ncnn
{

Mat Mat::clone(Allocator *_allocator) const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, _allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, _allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, _allocator);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * elemsize);
    }

    return m;
}

void Mat::clone_from(const ncnn::Mat &mat, Allocator *allocator)
{
    *this = mat.clone(allocator);
}

Mat Mat::reshape(int _w, Allocator *_allocator) const
{
    if (w * h * c != _w)
        return Mat();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Mat m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void *ptr = (unsigned char *)data + i * cstep * elemsize;
            void *mptr = (unsigned char *)m.data + (size_t)i * w * h * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

Mat Mat::reshape(int _w, int _h, Allocator *_allocator) const
{
    if (w * h * c != _w * _h)
        return Mat();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Mat m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void *ptr = (unsigned char *)data + i * cstep * elemsize;
            void *mptr = (unsigned char *)m.data + (size_t)i * w * h * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.cstep = (size_t)_w * _h;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _c, Allocator *_allocator) const
{
    if (w * h * c != _w * _h * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void *ptr = (unsigned char *)data + (size_t)i * _w * _h * elemsize;
                void *mptr = (unsigned char *)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * elemsize, 16) / elemsize;

    return m;
}

void Mat::create(int _w, size_t _elemsize, Allocator *_allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int *)(((unsigned char *)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, Allocator *_allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = (size_t)w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int *)(((unsigned char *)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator *_allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int *)(((unsigned char *)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator *_allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int *)(((unsigned char *)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator *_allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = (size_t)w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int *)(((unsigned char *)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator *_allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int *)(((unsigned char *)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create_like(const Mat &m, Allocator *_allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

Mat Mat::from_float16(const unsigned short *data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float *ptr = m; //.data;

    int remain = size;
    for (; remain > 0; remain--)
    {
        *ptr = float16_to_float32(*data);

        data++;
        ptr++;
    }

    return m;
}

} // namespace ncnn
