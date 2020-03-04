/* Copyright 2019-2020 Canaan Inc.
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
#include "transform.h"
#include <vector>

namespace nncase
{
class target;
}

namespace nncase::llir::transforms
{
class pass
{
public:
    void run(graph &graph, nncase::target &target);

    template <class T, class... TArgs>
    transform *emplace(TArgs &&... args)
    {
        return static_cast<T *>(transforms_.emplace_back(new T(std::forward<TArgs>(args)...)).get());
    }

private:
    std::vector<std::unique_ptr<transform>> transforms_;
};

class pass_manager
{
public:
    pass_manager(graph &graph, nncase::target &target)
        : graph_(graph), target_(target) {}

    void add_pass(pass &&pass) { passes_.emplace_back(std::move(pass)); }
    void run();

private:
    std::vector<pass> passes_;
    graph &graph_;
    nncase::target &target_;
};
}
