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
#include <iomanip>
#include <iostream>
#include <unordered_map>

class op_profile
{
public:
    op_profile() = default;
    op_profile(const std::string &op_type)
        : op_type_(op_type)
    {
        // begin_ = clock();
    }
    ~op_profile()
    {
        // end_ = clock();
        // auto cast_time = (end_ - begin_) / (double)1000;
        // if (op_time_cast_.find(op_type_) == op_time_cast_.end())
        // {
        //     op_time_cast_.emplace(op_type_, cast_time);
        // }
        // else
        // {
        //     op_time_cast_[op_type_] += cast_time;
        // }
    }
    void print_profile();

public:
    static std::unordered_map<std::string, double> op_time_cast_;

private:
    clock_t begin_;
    clock_t end_;
    std::string op_type_;
};