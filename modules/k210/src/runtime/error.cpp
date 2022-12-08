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
#include <nncase/runtime/k210/error.h>

using namespace nncase;
using namespace nncase::runtime::k210;

namespace {
class nncase_k210_error_category : public std::error_category {
  public:
    static nncase_k210_error_category instance;

    char const *name() const noexcept override { return "nncase_k210"; }

    std::string message(int code) const override {
        switch ((nncase_k210_errc)code) {
        case nncase_k210_errc::k210_illegal_instruction:
            return "K210 illegal instruction";
        default:
            return "Unknown nncase k210 error";
        }
    }

    bool equivalent(NNCASE_UNUSED std::error_code const &code,
                    NNCASE_UNUSED int condition) const noexcept override {
        return false;
    }
};

nncase_k210_error_category nncase_k210_error_category::instance;
} // namespace

const std::error_category &
nncase::runtime::k210::nncase_k210_category() noexcept {
    return nncase_k210_error_category::instance;
}

std::error_condition
nncase::runtime::k210::make_error_code(nncase_k210_errc code) {
    return std::error_code(static_cast<int>(code), nncase_k210_category());
}

std::error_condition
nncase::runtime::k210::make_error_condition(nncase_k210_errc code) {
    return std::error_condition(static_cast<int>(code), nncase_k210_category());
}
