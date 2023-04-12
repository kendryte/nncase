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
#include "vulkan_error.h"

using namespace nncase;
using namespace vk;

namespace {
class vulkan_error_category : public std::error_category {
  public:
    static vulkan_error_category instance;

    char const *name() const noexcept override { return "vulkan"; }

    std::string message(int code) const override {
        return vk::to_string((vk::Result)code);
    }

    bool equivalent(NNCASE_UNUSED std::error_code const &code,
                    NNCASE_UNUSED int condition) const noexcept override {
        return false;
    }
};

vulkan_error_category vulkan_error_category::instance;
} // namespace

const std::error_category &vk::vulkan_category() noexcept {
    return vulkan_error_category::instance;
}

std::error_condition vk::make_error_condition(vk::Result code) {
    return std::error_condition(static_cast<int>(code), vulkan_category());
}
