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

#include <nncase/runtime/stackvm/op_profile.h>

void op_profile::print_profile()
{
    std::cout << "OPS PROFILE" << std::endl;
    double ops_time = 0.;
    for (auto &it : op_time_cast_)
    {
        if (it.first == "")
            continue;
        std::cout << "Op type: " << std::setw(15) << std::left << it.first << "(" << it.second << " ms)" << std::endl;
        ops_time += it.second;
    }
    std::cout << "All OPS cast: " << ops_time << std::endl;
}