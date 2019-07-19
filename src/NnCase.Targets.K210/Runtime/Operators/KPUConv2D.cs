using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using NnCase.Runtime;
using NnCase.Targets.K210.IR;
using NnCase.Targets.K210.Runtime.Hardware;

namespace NnCase.Targets.K210.Runtime.Operators
{
    public class KPUConv2DOptions
    {
        public MemoryRange MainMemoryOutput { get; set; }

        public kpu_layer_argument_t LayerArgument;

        public kpu_batchnorm_argument_t[] BatchNorm { get; set; }

        public kpu_activate_table_t Activation { get; set; }

        public byte[] Weights { get; set; }
    }

    public class KPUConv2DOptionsBody : INodeBody
    {
        public RuntimeOpCode OpCode => RuntimeOpCode.K210_KPUConv2D;

        public KPUConv2DOptions Options { get; set; }

        public void Deserialize(ref MemoryReader reader)
        {
            Options = new KPUConv2DOptions
            {
                MainMemoryOutput = reader.Read<MemoryRange>(),
                LayerArgument = reader.Read<kpu_layer_argument_t>()
            };

            var ic = Options.LayerArgument.image_channel_num.i_ch_num + 1;
            var oc = Options.LayerArgument.image_channel_num.i_ch_num + 1;
            var filter = KPUShapeUtility.GetKPUFilterSize((KPUFilterType)(int)Options.LayerArgument.kernel_pool_type_cfg.kernel_type);
            var weights = Options.LayerArgument.interrupt_enabe.depth_wise_layer == 0
                ? ic * oc * filter * filter
                : oc * filter * filter;

            reader.Skip((int)(uint)Options.LayerArgument.kernel_pool_type_cfg.bwsx_base_addr);
            Options.BatchNorm = reader.ReadArray<kpu_batchnorm_argument_t>(Options.LayerArgument.image_channel_num.o_ch_num);
            reader.Skip((int)(uint)Options.LayerArgument.kernel_calc_type_cfg.active_addr);
            Options.Activation = new kpu_activate_table_t
            {
                activate_para = reader.ReadArray<activate_para_t>(16),
                activate_para_bias0 = reader.Read<activate_para_bias0_t>(),
                activate_para_bias1 = reader.Read<activate_para_bias1_t>()
            };
            reader.Skip((int)(uint)Options.LayerArgument.kernel_load_cfg.para_start_addr);
            Options.Weights = reader.ReadArray<byte>(weights);
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Options.MainMemoryOutput);

            var layerPos = writer.Position;
            writer.Position += Unsafe.SizeOf<kpu_layer_argument_t>();
            Options.LayerArgument.kernel_pool_type_cfg.bwsx_base_addr = (uint)writer.AlignPosition(8);
            writer.Write(Options.BatchNorm);
            Options.LayerArgument.kernel_calc_type_cfg.active_addr = (uint)writer.AlignPosition(256);
            writer.Write(Options.Activation.activate_para);
            writer.Write(Options.Activation.activate_para_bias0);
            writer.Write(Options.Activation.activate_para_bias1);
            Options.LayerArgument.kernel_load_cfg.para_start_addr = (uint)writer.AlignPosition(128);
            writer.Write(Options.Weights);

            var endPos = writer.Position;
            writer.Position = layerPos;
            writer.Write(Options.LayerArgument);
            writer.Position = endPos;
        }
    }
}
