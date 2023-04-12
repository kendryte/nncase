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
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>

using namespace nncase;
using namespace nncase::runtime;

namespace {
class nncase_error_category : public std::error_category {
  public:
    static nncase_error_category instance;

    char const *name() const noexcept override { return "nncase"; }

    std::string message(int code) const override {
        switch ((nncase_errc)code) {
        case nncase_errc::invalid_model_indentifier:
            return "Invalid model indentifier";
        case nncase_errc::invalid_model_checksum:
            return "Invalid model checksum";
        case nncase_errc::invalid_model_version:
            return "Invalid model version";
        case nncase_errc::runtime_not_found:
            return "Runtime not found";
        case nncase_errc::datatype_mismatch:
            return "Datatype mismatch";
        case nncase_errc::shape_mismatch:
            return "Shape mismatch";
        case nncase_errc::invalid_memory_location:
            return "Invalid memory location";
        case nncase_errc::stackvm_illegal_instruction:
            return "StackVM illegal instruction";
        case nncase_errc::stackvm_illegal_target:
            return "StackVM illegal target";
        case nncase_errc::stackvm_stack_overflow:
            return "StackVM stack overflow";
        case nncase_errc::stackvm_stack_underflow:
            return "StackVM stack underflow";
        case nncase_errc::nnil_illegal_instruction:
            return "NNIL illegal instruction";
        default:
            return "Unknown nncase error";
        }
    }

    bool equivalent(NNCASE_UNUSED std::error_code const &code,
                    NNCASE_UNUSED int condition) const noexcept override {
        return false;
    }
};

nncase_error_category nncase_error_category::instance;
} // namespace

const std::error_category &nncase::runtime::nncase_category() noexcept {
    return nncase_error_category::instance;
}

std::error_code nncase::runtime::make_error_code(nncase_errc code) {
    return std::error_code(static_cast<int>(code), nncase_category());
}

std::error_condition nncase::runtime::make_error_condition(nncase_errc code) {
    return std::error_condition(static_cast<int>(code), nncase_category());
}
