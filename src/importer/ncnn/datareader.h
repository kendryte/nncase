// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_DATAREADER_H
#define NCNN_DATAREADER_H

#include <stdio.h>

namespace ncnn {

// data read wrapper
class DataReader
{
public:
    DataReader();
    virtual ~DataReader();

    // parse plain param text
    // return 1 if scan success
    virtual int scan(const char* format, void* p) const;

    // read binary param and model data
    // return bytes read
    virtual size_t read(void* buf, size_t size) const;
};

class DataReaderFromStdioPrivate;
class DataReaderFromStdio : public DataReader
{
public:
    explicit DataReaderFromStdio(FILE* fp);
    virtual ~DataReaderFromStdio();

    virtual int scan(const char* format, void* p) const;
    virtual size_t read(void* buf, size_t size) const;

private:
    DataReaderFromStdio(const DataReaderFromStdio&);
    DataReaderFromStdio& operator=(const DataReaderFromStdio&);

private:
    DataReaderFromStdioPrivate* const d;
};

class DataReaderFromMemoryPrivate;
class DataReaderFromMemory : public DataReader
{
public:
    explicit DataReaderFromMemory(const unsigned char*& mem);
    virtual ~DataReaderFromMemory();

    virtual int scan(const char* format, void* p) const;
    virtual size_t read(void* buf, size_t size) const;

private:
    DataReaderFromMemory(const DataReaderFromMemory&);
    DataReaderFromMemory& operator=(const DataReaderFromMemory&);

private:
    DataReaderFromMemoryPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_DATAREADER_H
