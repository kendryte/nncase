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

#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include <nncase/ir/ir_types.h>

namespace ncnn
{

class Layer
{
public:
    std::string type;
    std::string name;
    std::vector<std::string> bottoms;
    std::vector<std::string> tops;

    // shape hint
    std::vector<nncase::ir::shape_t> bottom_shapes;
    std::vector<nncase::ir::shape_t> top_shapes;
};

} // namespace ncnn

#endif // NCNN_LAYER_H
