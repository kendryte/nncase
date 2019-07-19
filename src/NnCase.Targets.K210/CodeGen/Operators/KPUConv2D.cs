using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.Runtime;
using NnCase.Targets.K210.IR;
using NnCase.Targets.K210.IR.FakeOperators;
using NnCase.Targets.K210.Runtime.Hardware;
using NnCase.Targets.K210.Runtime.Operators;

namespace NnCase.Targets.K210.CodeGen.Operators
{
    internal static partial class K210Emitters
    {
        private static void RegisterKPUConv2D(CodeGenRegistry registry)
        {
            registry.Add<KPUConv2D>((n, g) =>
            {
                kpu_batchnorm_argument_t[] GetBatchNorm()
                {
                    return n.BatchNorm.Select(x => new kpu_batchnorm_argument_t
                    {
                        norm_mul = (uint)x.Mul,
                        norm_shift = (byte)x.Shift,
                        norm_add = (uint)x.Add
                    }).ToArray();
                }

                unsafe kpu_activate_table_t GetActivation()
                {
                    var table = new kpu_activate_table_t { activate_para = new activate_para_t[16] };
                    for (int i = 0; i < n.Activation.Length; i++)
                    {
                        ref var src = ref n.Activation[i];
                        table.activate_para[i] = new activate_para_t
                        {
                            x_start = (ulong)src.StartX,
                            y_mul = (ushort)src.Mul,
                            shift_number = (byte)src.Shift
                        };

                        if (i < 8)
                            table.activate_para_bias0.result_bias[i] = (byte)src.Add;
                        else
                            table.activate_para_bias1.result_bias[i - 8] = (byte)src.Add;
                    }

                    return table;
                }

                kpu_layer_argument_t GetLayer()
                {
                    var reg = new kpu_layer_argument_t();

                    int oneLoadKernelSize, loadTimes, ocOneTime;
                    if (n.IsDepthwise)
                    {
                        oneLoadKernelSize = (int)n.Weights.Length;
                        loadTimes = 1;
                        ocOneTime = n.KPUOutput.Shape[1];
                    }
                    else
                    {
                        var kernelSize = (int)n.Weights.Length;
                        var filter = KPUShapeUtility.GetKPUFilterSize(n.FilterType);
                        var oneChannelSize = filter * filter * n.Input.Shape[1];
                        var sizeLimit = 30;
                        var oneLoadChannels = Math.Min(n.KPUOutput.Shape[1], (int)Math.Floor(sizeLimit * 1024.0 / oneChannelSize));
                        oneLoadKernelSize = oneChannelSize * oneLoadChannels;
                        loadTimes = (int)Math.Ceiling(n.KPUOutput.Shape[1] / (double)oneLoadChannels);
                        ocOneTime = oneLoadChannels;
                    }

                    var inputLayout = K210Helper.GetRowLayout(n.Input.Shape[3]);
                    var outputLayout = K210Helper.GetRowLayout(n.KPUOutput.Shape[3]);

                    reg.interrupt_enabe = new interrupt_enabe_t
                    {
                        depth_wise_layer = (byte)(n.IsDepthwise ? 1 : 0)
                    };
                    reg.image_addr = new image_addr_t
                    {
                        image_src_addr = K210Helper.GetKpuAddress(g.MemoryRange(n.Input).Start),
                        image_dst_addr = K210Helper.GetKpuAddress(g.MemoryRange(n.KPUOutput).Start)
                    };
                    reg.image_channel_num = new image_channel_num_t
                    {
                        i_ch_num = (ushort)(n.Input.Shape[1] - 1),
                        o_ch_num = (ushort)(n.KPUOutput.Shape[1] - 1),
                        o_ch_num_coef = (ushort)(ocOneTime - 1)
                    };
                    reg.image_size = new image_size_t
                    {
                        i_row_wid = (ushort)(n.Input.Shape[3] - 1),
                        i_col_high = (ushort)(n.Input.Shape[2] - 1),
                        o_row_wid = (ushort)(n.KPUOutput.Shape[3] - 1),
                        o_col_high = (ushort)(n.KPUOutput.Shape[2] - 1)
                    };
                    reg.kernel_pool_type_cfg = new kernel_pool_type_cfg_t
                    {
                        load_para = 1,
                        kernel_type = (byte)n.FilterType,
                        pool_type = (byte)n.PoolType,
                        dma_burst_size = 15,
                        pad_value = n.PadValue
                    };
                    reg.kernel_load_cfg = new kernel_load_cfg_t
                    {
                        load_coor = 1,
                        load_time = (byte)(loadTimes - 1),
                        para_size = (uint)oneLoadKernelSize
                    };
                    reg.kernel_calc_type_cfg = new kernel_calc_type_cfg_t
                    {
                        channel_switch_addr = (ushort)(inputLayout.rowLength * n.Input.Shape[2]),
                        row_switch_addr = (byte)inputLayout.rowLength,
                        coef_group = (byte)inputLayout.groups,
                        load_act = 1
                    };
                    reg.write_back_cfg = new write_back_cfg_t
                    {
                        wb_channel_switch_addr = (ushort)(outputLayout.rowLength * n.KPUOutput.Shape[2]),
                        wb_row_switch_addr = (byte)outputLayout.rowLength,
                        wb_group = (byte)outputLayout.groups
                    };
                    reg.conv_value = new conv_value_t
                    {
                        shr_w = (byte)n.ShiftW,
                        shr_x = (byte)n.ShiftX,
                        arg_w = (uint)n.ArgW,
                        arg_x = (uint)n.ArgX
                    };
                    reg.conv_value2 = new conv_value2_t
                    {
                        arg_add = (ulong)n.ArgAdd
                    };
                    reg.dma_parameter = new dma_parameter_t
                    {
                        channel_byte_num = (ushort)(n.KPUOutput.Shape[3] * n.KPUOutput.Shape[2] - 1),
                        dma_total_byte = (uint)(n.KPUOutput.Shape[3] * n.KPUOutput.Shape[2] * n.KPUOutput.Shape[1] - 1),
                        send_data_out = n.MainMemoryOutput == null ? (byte)0 : (byte)1
                    };

                    return reg;
                }

                return new KPUConv2DOptionsBody
                {
                    Options = new KPUConv2DOptions
                    {
                        MainMemoryOutput = n.MainMemoryOutput == null ? new MemoryRange() : g.MemoryRange(n.MainMemoryOutput),
                        LayerArgument = GetLayer(),
                        BatchNorm = GetBatchNorm(),
                        Activation = GetActivation(),
                        Weights = n.Weights.ToArray()
                    }
                };
            });
        }
    }
}
