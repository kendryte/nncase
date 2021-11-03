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
#pragma once
#include "../node.h"

namespace nncase::ir
{
class NNCASE_API random_normal : public node
{
public:
    DEFINE_NODE_OPCODE(op_random_normal);

    output_connector &output() { return output_at(0); }

    float mean() const noexcept { return mean_; }
    float std() const noexcept { return std_; }
    float seed() const noexcept { return seed_; }
    random_normal(datatype_t output_type, shape_t output_shape, float mean, float std, float seed);

protected:
    bool properties_equal(node &other) const override;

private:
    float mean_;
    float std_;
    float seed_;
};
}
