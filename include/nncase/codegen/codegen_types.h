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
#include <iostream>
#include <nncase/codegen/binary_writer.h>
#include <nncase/ir/node.h>
#include <nncase/runtime/datatypes.h>

namespace nncase::codegen
{
class section_writer;

struct symbol_ref
{
    std::string name;
    std::streampos streampos;
    size_t bitoffset;
    size_t length;
};

struct symbol
{
    std::string name;
    std::streampos streampos;
};

class section_writer : public binary_writer
{
public:
    using binary_writer::binary_writer;

    NNCASE_API std::span<const symbol> symbols() const noexcept { return symbols_; }
    NNCASE_API std::span<const symbol_ref> symbol_refs() const noexcept { return symbol_refs_; }

    NNCASE_API void add_symbol_ref(size_t offset, size_t length, std::string_view name)
    {
        symbol_refs_.emplace_back(symbol_ref { std::string(name), position(), offset, length });
    }

    NNCASE_API void add_symbol(std::string_view name)
    {
        auto pos = position();
        symbols_.emplace_back(symbol { std::string(name), pos });
    }

private:
    std::vector<symbol_ref> symbol_refs_;
    std::vector<symbol> symbols_;
};
}
