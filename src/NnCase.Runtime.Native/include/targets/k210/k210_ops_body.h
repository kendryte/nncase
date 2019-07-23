#pragma once
#include "../node_body.h"
#include "k210_runtime_op_utility.h"
#include "k210_sim_types.h"

namespace nncase
{
namespace targets
{
    namespace k210
    {
        struct kpu_upload_options : simple_node_body<kpu_upload_options>
        {
            memory_range input;
            memory_range output;
            runtime_shape_t in_shape;
        };

        struct kpu_conv2d_options
        {
            memory_range main_mem_output;
            int32_t batches;
            int32_t reserved0;
            kpu_layer_argument_t layer;
            xtl::span<const kpu_batchnorm_argument_t> batch_norm;
            const kpu_activate_table_t *activation;
            xtl::span<const uint8_t> weights;

            void deserialize(runtime::span_reader &reader)
            {
                reader.read(main_mem_output);
                reader.read(batches);
                reader.read(reserved0);
                reader.read(layer);

                auto ic = layer.image_channel_num.data.i_ch_num + 1;
                auto oc = layer.image_channel_num.data.o_ch_num + 1;
                auto filter = get_kpu_filter_size((kpu_filter_type_t)layer.kernel_pool_type_cfg.data.kernel_type);
                auto weights_size = layer.interrupt_enabe.data.depth_wise_layer
                    ? oc * filter * filter
                    : ic * oc * filter * filter;

                reader.skip(layer.kernel_pool_type_cfg.data.bwsx_base_addr);
                reader.read_span(batch_norm, oc);
                reader.skip(layer.kernel_calc_type_cfg.data.active_addr);
                reader.get_ref(activation);
                reader.skip(layer.kernel_load_cfg.data.para_start_addr);
                reader.read_span(weights, weights_size);
#if !NNCASE_TARGET_K210_SIMULATOR
                layer.kernel_pool_type_cfg.data.bwsx_base_addr = (uintptr_t)batch_norm.data();
                layer.kernel_calc_type_cfg.data.active_addr = (uintptr_t)activation;
                layer.kernel_load_cfg.data.para_start_addr = (uintptr_t)weights.data();
#endif
            }
        };
    }
}
}
