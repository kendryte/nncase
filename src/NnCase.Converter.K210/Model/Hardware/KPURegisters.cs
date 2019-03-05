using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Converter.K210.Model.Hardware
{
    public partial struct interrupt_enabe_t
    {
        enum BitFields { int_en = 1, ram_flag = 1, full_add = 1, depth_wise_layer = 1, reserved = 60 }
    }

    public partial struct image_addr_t
    {
        enum BitFields { image_src_addr = 15, reserved0 = 17, image_dst_addr = 15, reserved1 = 17 }
    }

    public partial struct image_channel_num_t
    {
        enum BitFields { i_ch_num = 10, reserved0 = 22, o_ch_num = 10, reserved1 = 6, o_ch_num_coef = 10, reserved2 = 6 }
    }

    public partial struct image_size_t
    {
        enum BitFields { i_row_wid = 10, i_col_high = 9, reserved0 = 13, o_row_wid = 10, o_col_high = 9, reserved1 = 13 }
    }

    public partial struct kernel_pool_type_cfg_t
    {
        enum BitFields { kernel_type = 3, pad_type = 1, pool_type = 4, first_stride = 1, bypass_conv = 1, load_para = 1, reserved0 = 5, dma_burst_size = 8, pad_value = 8, bwsx_base_addr = 32 }
    }

    public partial struct kernel_load_cfg_t
    {
        enum BitFields { load_coor = 1, load_time = 6, reserved0 = 8, para_size = 17, para_start_addr = 32 }
    }

    public partial struct kernel_offset_t
    {
        enum BitFields { coef_column_offset = 4, coef_row_offset = 12, reserved0 = 40 }
    }

    public partial struct kernel_calc_type_cfg_t
    {
        enum BitFields { channel_switch_addr = 15, reserved = 1, row_switch_addr = 4, coef_size = 8, coef_group = 3, load_act = 1, active_addr = 32 }
    }

    public partial struct write_back_cfg_t
    {
        enum BitFields { wb_channel_switch_addr = 15, reserved0 = 1, wb_row_switch_addr = 4, wb_group = 3, reserved1 = 41 }
    }

    public partial struct conv_value_t
    {
        enum BitFields { shr_w = 4, shr_x = 4, arg_w = 24, arg_x = 24, reserved0 = 8 }
    }

    public partial struct conv_value2_t
    {
        enum BitFields { arg_add = 40, reserved = 24 }
    }

    public partial struct dma_parameter_t
    {
        enum BitFields { send_data_out = 1, reserved = 15, channel_byte_num = 16, dma_total_byte = 32 }
    }

    public struct kpu_layer_argument_t
    {
        public interrupt_enabe_t interrupt_enabe;
        public image_addr_t image_addr;
        public image_channel_num_t image_channel_num;
        public image_size_t image_size;
        public kernel_pool_type_cfg_t kernel_pool_type_cfg;
        public kernel_load_cfg_t kernel_load_cfg;
        public kernel_offset_t kernel_offset;
        public kernel_calc_type_cfg_t kernel_calc_type_cfg;
        public write_back_cfg_t write_back_cfg;
        public conv_value_t conv_value;
        public conv_value2_t conv_value2;
        public dma_parameter_t dma_parameter;
    }

    public partial struct activate_para_t
    {
        enum BitFields { shift_number = 8, y_mul = 16, x_start = 36 }
    }

    [StructLayout(LayoutKind.Explicit, Pack = 1, Size = 8)]
    public struct activate_para_bias0_t
    {
        [FieldOffset(0)]
        public ulong Value;

        [FieldOffset(0)]
        public unsafe fixed byte result_bias[8];
    }

    [StructLayout(LayoutKind.Explicit, Pack = 1, Size = 8)]
    public struct activate_para_bias1_t
    {
        [FieldOffset(0)]
        public ulong Value;

        [FieldOffset(0)]
        public unsafe fixed byte result_bias[8];
    }

    public class kpu_activate_table_t
    {
        public activate_para_t[] activate_para { get; } = new activate_para_t[16];

        public activate_para_bias0_t activate_para_bias0;
        public activate_para_bias1_t activate_para_bias1;
    }

    public partial struct kpu_batchnorm_argument_t
    {
        enum BitFields { norm_mul = 24, norm_add = 32, norm_shift = 4 }
    }
}
