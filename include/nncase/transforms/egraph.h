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
#include <list>
#include <nncase/ir/function.h>

namespace nncase::ir::transforms {
using eid_t = size_t;

class NNCASE_API enode {
  public:
    /** @brief Get the expression of the enode */
    const ir::expr &expr() const noexcept { return expr_; }
    /** @brief Get the mutable expression of the enode */
    ir::expr &expr() noexcept { return expr_; }
    /** @brief Set the expression of the enode */
    void expr(ir::expr value) noexcept { expr_ = std::move(value); }

    /** @brief Get the children eids of the enode */
    const std::vector<eid_t> &children() const noexcept { return children_; }
    /** @brief Get the mutable children eids of the enode */
    std::vector<eid_t> &children() noexcept { return children_; }

  private:
    ir::expr expr_;
    std::vector<eid_t> children_;
};

class NNCASE_API eclass {
  public:
    /** @brief Get the enodes of the eclass */
    const std::vector<enode *> &nodes() const noexcept { return nodes_; }
    /** @brief Get the mutable enodes of the eclass */
    std::vector<enode *> &nodes() noexcept { return nodes_; }

  private:
    eid_t id_;
    std::vector<enode *> nodes_;
};

class NNCASE_API egraph {
  private:
    std::list<eclass> classes_;
    std::list<enode> nodes_;
};
} // namespace nncase::ir::transforms
