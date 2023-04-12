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
#include <nncase/kernels/k210/k210_kernels.h>
#ifndef NNCASE_SIMULATOR
#include <dmac.h>
#include <kpu.h>
#include <sysctl.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

result<void> nncase::kernels::k210::kpu_upload(const uint8_t *src,
                                               uint8_t *dest,
                                               const kpu_shape_t &in_shape,
                                               NNCASE_UNUSED uint32_t dma_ch) {
    if (in_shape[3] % 64 == 0) {
        auto size_bytes = kernels::detail::compute_size(in_shape);
#ifdef NNCASE_SIMULATOR
        std::copy(src, src + size_bytes, dest);
#else
        auto ch = (dmac_channel_number_t)dma_ch;
        dmac_set_irq(ch, nullptr, nullptr, 1);
        dmac_set_single_mode(ch, (void *)(src - IOMEM), (void *)dest,
                             DMAC_ADDR_INCREMENT, DMAC_ADDR_INCREMENT,
                             DMAC_MSIZE_16, DMAC_TRANS_WIDTH_64,
                             size_bytes / 8);
        dmac_wait_done(ch);
#endif
    } else {
        auto layout = get_kpu_row_layout(in_shape[3]);
        auto fmap_size = get_kpu_bytes(in_shape[3], in_shape[2], in_shape[1]);

        for (uint32_t batch = 0; batch < in_shape[0]; batch++) {
            auto batch_origin = dest + (size_t)batch * fmap_size;
            for (uint32_t oc = 0; oc < in_shape[1]; oc++) {
                auto channel_origin =
                    batch_origin +
                    (size_t)oc / layout.groups * layout.row_len * in_shape[2] *
                        64 +
                    (size_t)oc % layout.groups * layout.row_pitch;
                for (uint32_t y = 0; y < in_shape[2]; y++) {
                    auto y_origin =
                        channel_origin + (size_t)y * layout.row_len * 64;
                    std::copy(src, src + in_shape[3], y_origin);
                    src += in_shape[3];
                }
            }
        }
    }

    return ok();
}
