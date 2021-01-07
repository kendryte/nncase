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
#include <nncase/runtime/error.h>

using namespace nncase;
using namespace nncase::runtime;

namespace
{
class nncase_error_category : public std::error_category
{
public:
    static nncase_error_category instance;

    char const *name() const noexcept override
    {
        return "nncase";
    }

    std::string message(int code) const override
    {
        return "Message"; // Íµ¸öÀÁ
    }

    bool equivalent(std::error_code const &code, int condition) const noexcept override
    {
        return false;
    }
};

nncase_error_category nncase_error_category::instance;
}

const std::error_category &nncase::runtime::nncase_category() noexcept
{
    return nncase_error_category::instance;
}

std::error_condition nncase::runtime::make_error_condition(nncase_errc code)
{
    return std::error_condition(static_cast<int>(code), nncase_category());
}
