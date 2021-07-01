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
#include "runtime_types.h"
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>

BEGIN_NS_NNCASE_RT_K210

class NNCASE_MODULES_K210_API op_visitor
{
public:
    op_visitor() noexcept
        : reader_({})
    {
    }

    ~op_visitor() = default;

    result<void> visit(gsl::span<const gsl::byte> text) noexcept;

    virtual result<void> visit(NNCASE_UNUSED const kpu_download_options &op) noexcept { return ok(); }
    virtual result<void> visit(NNCASE_UNUSED const kpu_conv2d_options &op) noexcept { return ok(); }
    virtual result<void> visit(NNCASE_UNUSED const kpu_upload_options &op) noexcept { return ok(); }
    virtual result<void> visit(NNCASE_UNUSED const copy_options &op) noexcept { return ok(); }

protected:
    bool interrupted_;
    span_reader reader_;

private:
    result<void> next() noexcept;
};

END_NS_NNCASE_RT_K210