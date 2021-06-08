/* Copyright 2020 Canaan Inc.
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
#include "tflite/tflite_importer.h"
#include "onnx/onnx_importer.h"
#include "caffe/caffe_importer.h"
#include <nncase/importer/importer.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

void nncase::importer::import_tflite(ir::graph &graph, std::span<const uint8_t> model, const import_options &options)
{
    tflite_importer(model, graph).import(options);
}

void nncase::importer::import_onnx(ir::graph &graph, std::span<const uint8_t> model, const import_options &options)
{
    onnx_importer(model, graph).import(options);
}

void nncase::importer::import_caffe(ir::graph &graph, std::span<const uint8_t> model, const import_options &options)
{
    caffe_importer(model, graph).import(options);
}