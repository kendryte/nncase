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

#if defined(__riscv)

#define RISCVFREQUENCY 1600000000

static uint64_t k230_get_cycles()
{
    uint64_t x;
    __asm volatile("rdcycle %0;"
                   : "=r"(x)::);
    return x;
}
#endif

class op_profile
{
public:
    op_profile(const std::string &op_type = "op_profile")
        : op_type_(op_type)
    {
#if defined(__riscv)
        begin_ = k230_get_cycles();
#else
        begin_ = clock();
#endif
    }

    ~op_profile()
    {
#if defined(__riscv)

        end_ = k230_get_cycles();
        auto cast_time = end_ - begin_;
        // std::cout << "cpu op:" << op_type_ << " cast time:" << cast_time << " begin time:" << begin_ << " end time:" << end_ << " " << std::endl;
#else
        end_ = clock();
        auto cast_time = (end_ - begin_) / (double)1000;
        // std::cout << "cpu op:" << op_type_ << " cast time:" << cast_time << " begin time:" << begin_ << " end time:" << end_ << " " <<std::endl;
#endif
        if (op_timing_.find(op_type_) == op_timing_.end())
        {
            op_timing_.emplace(op_type_, cast_time);
        }
        else
        {
            op_timing_[op_type_] += cast_time;
        }
    }

    static void print();

public:
    static std::unordered_map<std::string, double> op_timing_;

private:
    clock_t begin_;
    clock_t end_;
    std::string op_type_;
};