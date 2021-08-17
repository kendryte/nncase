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
#include "ncnn_importer.h"
#include <algorithm>
#include <nncase/importer/importer.h>
#include <nncase/importer/util.h>
#include <nncase/ir/graph.h>
#include "datareader.h"
#include "modelbin.h"

using namespace std;
using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

ncnn_importer::ncnn_importer(const std::filesystem::path &_paramfilename, const std::filesystem::path &_binfilename, ir::graph &graph)
    : graph_(graph)
{
    paramfilename = _paramfilename;
    binfilename = _binfilename;
}

class FileWrapper
{
public:
    FileWrapper(const std::string& path)
    {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp)
            throw std::runtime_error("Cannot open file: " + path);
    }

    ~FileWrapper()
    {
        fclose(fp);
    }

    FILE* fp;
};

void ncnn_importer::import(const import_options &/*options*/)
{
    // load param
    FileWrapper paramfile(paramfilename.string());
    FileWrapper binfile(binfilename.string());

    ncnn::DataReaderFromStdio dr(paramfile.fp);
    ncnn::DataReaderFromStdio bindr(binfile.fp);

    ncnn::ModelBinFromDataReader mb(bindr);

#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        throw std::runtime_error("parse " #v " failed"); \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        throw std::runtime_error("param is too old, please regenerate");
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        throw std::runtime_error("invalid layer_count or blob_count");
    }

    std::unordered_map<std::string, ir::shape_t> shapes;

    ncnn::ParamDict pd;

    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        ncnn::Layer layer;
        layer.type = std::string(layer_type);
        layer.name = std::string(layer_name);

        layer.bottoms.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)
            layer.bottoms[j] = std::string(blob_name);
        }

        layer.tops.resize(bottom_count);
        for (int j = 0; j < top_count; j++)
        {
            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)
            layer.tops[j] = std::string(blob_name);
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            throw std::runtime_error("ParamDict load_param failed " + layer.name);
            continue;
        }

        // pull out top shape hints
        ncnn::Mat shape_hints = pd.get(30, ncnn::Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j = 0; j < top_count; j++)
            {
                ir::shape_t shape;

                int dims = psh[0];
                if (dims == 1)
                {
                    shape = shape_t{ (size_t)psh[1] };
                }
                if (dims == 2)
                {
                    shape = shape_t{ (size_t)psh[1], (size_t)psh[2] };
                }
                if (dims == 3)
                {
                    shape = shape_t{ (size_t)psh[1], (size_t)psh[2], (size_t)psh[3] };
                }

                shapes[layer.tops[j]] = shape;

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer.bottom_shapes.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            layer.bottom_shapes[j] = shapes[layer.bottoms[j]];
        }

        layer.top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            layer.top_shapes[j] = shapes[layer.tops[j]];
        }

        convert_op(layer, pd, mb);
    }

#undef SCAN_VALUE

    // connect tensors
    for (auto &&in : input_tensors_)
    {
        auto out_it = output_tensors_.find(in.second);
        if (out_it != output_tensors_.end())
        {
            in.first->connect(*out_it->second);
        }
        else
        {
            assert(!"Cannot find associated output node");
        }
    }

    // outputs
    for (auto &&out : output_tensors_)
    {
        if (out.second->connections().empty())
        {
            auto node = graph_.emplace<output_node>(out.second->type(), out.second->shape());
            node->name(out.first);
            out.second->connect(node->input());
        }
    }
}

void ncnn_importer::convert_op(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin& mb)
{
#define DEFINE_OPCODE(opcode) \
    if (layer.type == #opcode##sv)  \
        return convert_op_##opcode(layer, pd, mb);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error("Not supported ncnn opcode: " + layer.type);
}
