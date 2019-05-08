using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.K210.Emulator;
using NnCase.Converter.K210.Model.Hardware;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model.Layers;
using QuantizationRange = NnCase.Converter.K210.Converters.Stages.Quantize.QuantizationRange;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class K210Conv2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public K210Conv2dParamAddress ParamAddress { get; set; }

        public K210ConvLayerConfig Config { get; set; }
    }

    [LayerConverter(typeof(K210Conv2d), K210LayerType.K210Conv)]
    public class K210Conv2dConverter
    {
        public K210Conv2dLayerArgument Convert(K210Conv2d layer, ConvertContext context)
        {
            var config = new K210ConvLayerConfig { BNConfigs = new K210LayerBNConfig[layer.OutputChannels], ActConfigs = new K210LayerActConfig[16] };
            (var sw, var bw) = QuantizeWeights(layer.Conv2dType == K210Conv2dType.Conv2d, layer.Weights, config, context.WeightsBits);
            (var sx, var bx) = QuantizeInput(context.Quantization.Distributions[layer.Input.Connection.From].Global, config);
            config.ArgAdd = (long)Math.Round(bw * bx * layer.KernelWidth * layer.KernelHeight);

            var scale = new double[layer.OutputChannels];
            for (int i = 0; i < scale.Length; i++)
                scale[i] = sw[i] * sx;

            QuantizeBiasAndOutput(layer, layer.Bias, context.Quantization.Distributions[layer.Output], context.Quantization.AdditionalDistributions[layer.OutputBeforeActivation], scale, config);

            config.InputChannels = layer.InputChannels;
            config.OutputChannels = layer.OutputChannels;

            config.InputWidth = layer.Input.Dimensions[3];
            config.InputHeight = layer.Input.Dimensions[2];
            (config.InputGroups, config.InputRowLength, _) = K210Helper.GetRowLayout(config.InputWidth);
            config.OutputWidth = layer.Output.Dimensions[3];
            config.OutputHeight = layer.Output.Dimensions[2];
            (config.OutputGroups, config.OutputRowLength, _) = K210Helper.GetRowLayout(config.OutputWidth);

            config.KernelType = layer.KernelWidth == 3 ? 1 : 0;
            config.IsDepthwise = layer.Conv2dType == K210Conv2dType.DepthwiseConv2d;
            config.PoolType = (int)layer.PoolType;

            config.PadValue = (int)Math.Round(-bx);

            if (layer.Conv2dType == K210Conv2dType.Conv2d)
            {
                var kernelSize = (int)layer.Weights.Length * context.WeightsBits / 8;
                var oneChannelSize = layer.KernelWidth * layer.KernelHeight * layer.InputChannels * context.WeightsBits / 8;
                var sizeLimit = context.WeightsBits == 8 ? 30 : 60;
                var oneLoadChannels = Math.Min(layer.OutputChannels, (int)Math.Floor(sizeLimit * 1024.0 / oneChannelSize));
                config.OneLoadKernelsSize = oneChannelSize * oneLoadChannels;
                config.LoadTimes = (int)Math.Ceiling(layer.OutputChannels / (double)oneLoadChannels);
                config.OutputChannelsOnTime = oneLoadChannels;
            }
            else
            {
                config.OneLoadKernelsSize = (int)layer.Weights.Length * context.WeightsBits / 8;
                config.LoadTimes = 1;
                config.OutputChannelsOnTime = layer.OutputChannels;
            }

            var inputOneLineChannels = Math.Min(layer.InputChannels, config.InputGroups);
            config.InputSize = config.InputRowLength * config.InputHeight * config.InputChannels / inputOneLineChannels;
            var outputOneLineChannels = Math.Min(layer.OutputChannels, config.OutputGroups);
            config.OutputSize = config.OutputRowLength * config.OutputHeight * config.OutputChannels / outputOneLineChannels;

            return new K210Conv2dLayerArgument
            {
                Config = config,
                ParamAddress = new K210Conv2dParamAddress()
            };
        }

        public void Infer(K210Conv2d layer, K210Conv2dLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.KPUMemoryMap[layer.Input.Connection.From];
            MemoryAllocation outputAlloc;

            argument.Config.InputAddress = inputAlloc.GetAddress();

            if (context.MainMemoryMap.TryGetValue(layer.Output, out var mainAlloc))
            {
                argument.Flags = K210LayerFlags.MainMemoryOutput;
                argument.MainMemoryOutputAddress = mainAlloc.GetAddress();
                outputAlloc = context.GetOrAllocateKPUMemory(layer.Output);
            }
            else
            {
                argument.Flags = K210LayerFlags.None;
                outputAlloc = context.KPUMemoryMap[layer.Output];
            }

            argument.Config.OutputAddress = outputAlloc.GetAddress();
        }

        public void GenerateBin(BinaryWriter bw, K210Conv2dLayerArgument argument, K210BinGenerationContext context)
        {
            bw.Write((uint)argument.Flags);
            bw.Write(argument.MainMemoryOutputAddress);
            // Param addresses
            var fixPosition = bw.BaseStream.Position;
            bw.BaseStream.Position += 4 * 4;

            GenerateBinLayer(bw, argument.Config, argument.ParamAddress, context);
            GenerateBinWeights(bw, argument.Config, argument.ParamAddress, context);
            GenerateBinBn(bw, argument.Config, argument.ParamAddress, context);
            GenerateBinActivation(bw, argument.Config, argument.ParamAddress, context);

            var newPosition = bw.BaseStream.Position;
            bw.BaseStream.Position = fixPosition;
            bw.Write(argument.ParamAddress.Layer);
            bw.Write(argument.ParamAddress.Weights);
            bw.Write(argument.ParamAddress.Bn);
            bw.Write(argument.ParamAddress.Activation);
            bw.BaseStream.Position = newPosition;
        }

        public K210Conv2dLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new K210Conv2dLayerArgument();
            argument.Flags = sr.Read<K210LayerFlags>();
            argument.MainMemoryOutputAddress = sr.Read<uint>();
            argument.ParamAddress = new K210Conv2dParamAddress
            {
                Layer = sr.Read<uint>(),
                Weights = sr.Read<uint>(),
                Bn = sr.Read<uint>(),
                Activation = sr.Read<uint>()
            };

            DeserializeBinLayer(argument, context);
            DeserializeBinWeights(argument, context);
            DeserializeBinBn(argument, context);
            DeserializeBinActivation(argument, context);

            return argument;
        }

        public static (double[] scale, double bias) QuantizeWeights(bool isConv2d, Tensor<float> weights, K210ConvLayerConfig config, int weightsBits)
        {
#if CHANNEL_WISE
            var kernels = weights.ToDenseTensor().Buffer.Span;
            var channels = weights.Dimensions[isConv2d ? 0 : 1];
            var channelSize = weights.Dimensions.GetSize() / channels;

            var totalRange = Quantizer.GetRange(kernels);
            var scales = new double[channels];

            for (int i = 0; i < channels; i++)
            {
                double s;
                var buffer = kernels.Slice(i * channelSize, channelSize);
                var range = Quantizer.GetRange(buffer);

                var s1 = totalRange.Max / range.Max;
                var s2 = totalRange.Min / range.Min;
                s = (s1 < 0 || s2 < 0) ? Math.Max(s1, s2) : Math.Min(s1, s2);

                Debug.Assert(s > 0);
                for (int j = 0; j < buffer.Length; j++)
                    buffer[j] = (float)(buffer[j] * s);
                scales[i] = s;
            }

            (var scale, var bias) = Quantizer.GetRange(kernels).GetScaleBias(weightsBits);

            (var mul, var shift) = Quantizer.ExtractValueAndShift(bias, 24, 15);
            config.Weights = Quantizer.Quantize(kernels, scale, bias, weightsBits);
            config.ArgX = (int)Math.Round(mul);
            config.ShiftX = shift;

            for (int i = 0; i < scales.Length; i++)
                scales[i] *= scale;
            return (scales, bias);
#else
            var buffer = weights.ToDenseTensor().Buffer.Span;
            (var scale, var bias) = GetRange(buffer).GetScaleBias();

            (var mul, var shift) = ExtractValueAndShift(bias, 24, 15);
            config.Weights = Quantize(buffer, scale, bias);
            config.ArgX = (int)Math.Round(mul);
            config.ShiftX = shift;
            return (Enumerable.Repeat(scale, weights.Dimensions[0]).ToArray(), bias);
#endif
        }

        private static (double scale, double bias) QuantizeInput(QuantizationRange range, K210ConvLayerConfig config)
        {
            (var scale, var bias) = range.GetScaleBias(8);
            (var mul, var shift) = Quantizer.ExtractValueAndShift(bias, 24, 15);
            config.ArgW = (int)Math.Round(mul);
            config.ShiftW = shift;
            return (scale, bias);
        }

        private static void QuantizeBiasAndOutput(K210Conv2d layer, Tensor<float> bias, ChannelwiseRange range, ChannelwiseRange beforeActRange, double[] scale, K210ConvLayerConfig config)
        {
            var upshift = 10;
            var postMul = Math.Pow(2, upshift);

            if (layer.IsChannelwiseOutput)
            {
                for (int i = 0; i < config.BNConfigs.Length; i++)
                {
                    (var so, var bo) = range.Channels[i].GetScaleBias(8);

                    var b = bias[i];

                    var scomb = so * postMul / scale[i];

                    (var mul, var shift) = Quantizer.ExtractValueAndShift(scomb, 22, 15);

                    config.BNConfigs[i] = new K210LayerBNConfig
                    {
                        Mul = (int)Math.Round(mul),
                        Shift = shift,
                        Add = (int)Math.Round((b * so - bo) * postMul)
                    };
                }
            }
            else
            {
                (var so, var bo) = range.Global.GetScaleBias(8);
#if CHANNEL_WISE

                for (int i = 0; i < config.BNConfigs.Length; i++)
                {
                    var b = bias[i];

                    var scomb = so * postMul / scale[i];

                    (var mul, var shift) = Quantizer.ExtractValueAndShift(scomb, 22, 15);

                    config.BNConfigs[i] = new K210LayerBNConfig
                    {
                        Mul = (int)Math.Round(mul),
                        Shift = shift,
                        Add = (int)Math.Round((b * so - bo) * postMul)
                    };
                }
#else
            var scomb = so / scale[0];

            (var mul, var shift) = ExtractValueAndShift(scomb, 22, 255);
            var upscale = shift - 15;
            Debug.Assert(upscale >= 0);
            var postMul = Math.Round(mul) / mul * Math.Pow(2, upscale);

            for (int i = 0; i < config.BNConfigs.Length; i++)
            {
                var b = bias[i];

                config.BNConfigs[i] = new K210LayerBNConfig
                {
                    Mul = (int)Math.Round(mul),
                    Shift = 15,
                    Add = (int)Math.Round((b * so - bo) * postMul)
                };
            }
#endif
            }

            QuantizeActivation(layer, postMul, range.Global, beforeActRange.Global, config);
        }

        private static void QuantizeActivation(K210Conv2d layer, double postMul, QuantizationRange range, QuantizationRange beforeActRange, K210ConvLayerConfig config)
        {
            if (layer.NonTrivialActivation == null)
            {
                switch (layer.FusedActivationFunction)
                {
                    case ActivationFunctionType.Linear:
                    case ActivationFunctionType.Relu:
                    case ActivationFunctionType.Relu6:
                        break;
                    default:
                        throw new NotSupportedException($"Activation of {layer.FusedActivationFunction} is not supported.");
                }

                var starts = new long[]
                {
                    0x800000000, 0xf7d4cf4b8, 0xf8ed5a20c, 0xfa05e4f60,
                    0xfb2e05baa, 0xfc46908fe, 0xfd5f1b652, 0xfe77a63a6,
                    0xff9fc6ff0, 0xfffd4a9b7, 0, 0x7FFFFFFF0,
                    0x7FFFFFFF1, 0x7FFFFFFF2, 0x7FFFFFFF3, 0x7FFFFFFF4
                };

                for (int i = 0; i < starts.Length; i++)
                {
                    var param = config.ActConfigs[i] = new K210LayerActConfig();
                    param.StartX = starts[i];

                    if (i == 10)
                    {
                        (var mul, var shift) = Quantizer.ExtractValueAndShift(1 / postMul, 16, 20);
                        param.Mul = (int)Math.Round(mul);
                        param.Shift = shift;
                    }
                }
            }
            else if (layer.NonTrivialActivation is LeakyRelu leakyRelu)
            {
                (var scale, var bias) = range.GetScaleBias(8);
                var zero = (long)(Quantizer.Quantize(0, scale, bias) * postMul);
                var yTable = Generator.IntegerStep(0, (int)-bias, 15).Take(14).ToArray();

                for (int i = 0; i < 16; i++)
                {
                    var param = config.ActConfigs[i] = new K210LayerActConfig();
                    if (i == 0)
                    {
                        param.StartX = 0x800000000;
                    }
                    else if (i == 15)
                    {
                        (var mul, var shift) = Quantizer.ExtractValueAndShift(1 / postMul, 16, 20);
                        param.StartX = zero;
                        param.Mul = (int)Math.Round(mul);
                        param.Shift = shift;
                        param.Add = (byte)(-bias);
                    }
                    else
                    {
                        // f(x) = (1 - slope) * zero + x * slope
                        // f(x1) - f(x0) = (x1 - x0) * slope
                        // x0 = zero - (zero - y0) / slope
                        var add = (byte)yTable[i - 1];
                        var y0 = add * postMul;
                        var x0 = zero - (zero - y0) / leakyRelu.Slope;

                        (var mul, var shift) = Quantizer.ExtractValueAndShift(1 / postMul * leakyRelu.Slope, 16, 20);
                        param.StartX = (long)Math.Floor(x0);
                        param.Mul = (int)Math.Round(mul);
                        param.Shift = shift;
                        param.Add = add;
                    }
                }
            }
            else
            {
                throw new NotSupportedException($"Activation of {layer.NonTrivialActivation.GetType().Name} is not supported.");
            }
        }

        private void GenerateBinWeights(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress, K210BinGenerationContext context)
        {
            paramAddress.Weights = context.AlignStreamPosition(128);

            if (context.WeightsBits == 8)
            {
                foreach (var v in layer.Weights)
                    bw.Write((byte)v);
            }
            else
            {
                foreach (var v in layer.Weights)
                    bw.Write(v);
            }
        }

        private void GenerateBinBn(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress, K210BinGenerationContext context)
        {
            paramAddress.Bn = context.AlignStreamPosition(128);

            for (int j = 0; j < layer.BNConfigs.Length; j++)
            {
                var bn = layer.BNConfigs[j];
                var reg = new kpu_batchnorm_argument_t();
                reg.norm_add = (uint)bn.Add;
                reg.norm_mul = (uint)bn.Mul;
                reg.norm_shift = (byte)bn.Shift;

                bw.Write(reg.Value);
            }
        }

        private void GenerateBinActivation(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress, K210BinGenerationContext context)
        {
            paramAddress.Activation = context.AlignStreamPosition(256);

            var reg = new kpu_activate_table_t();
            var configs = layer.ActConfigs;

            for (int i = 0; i < configs.Length; i++)
            {
                var config = configs[i];
                ref var param = ref reg.activate_para[i];
                param.x_start = (ulong)config.StartX;
                param.y_mul = (ushort)config.Mul;
                param.shift_number = (byte)config.Shift;
            }

            for (int i = 0; i < configs.Length; i++)
            {
                var config = configs[i];
                unsafe
                {
                    if (i < 8)
                        reg.activate_para_bias0.result_bias[i] = (byte)config.Add;
                    else
                        reg.activate_para_bias1.result_bias[i - 8] = (byte)config.Add;
                }
            }

            for (int j = 0; j < configs.Length; j++)
                bw.Write(reg.activate_para[j].Value);
            bw.Write(reg.activate_para_bias0.Value);
            bw.Write(reg.activate_para_bias1.Value);
        }

        private void GenerateBinLayer(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress, K210BinGenerationContext context)
        {
            paramAddress.Layer = context.AlignStreamPosition(8);

            var reg = new kpu_layer_argument_t();

            reg.interrupt_enabe = new interrupt_enabe_t
            {
                depth_wise_layer = (byte)(layer.IsDepthwise ? 1 : 0)
            };
            reg.image_addr = new image_addr_t
            {
                image_src_addr = (ushort)layer.InputAddress,
                image_dst_addr = (ushort)layer.OutputAddress
            };
            reg.image_channel_num = new image_channel_num_t
            {
                i_ch_num = (ushort)(layer.InputChannels - 1),
                o_ch_num = (ushort)(layer.OutputChannels - 1),
                o_ch_num_coef = (ushort)(layer.OutputChannelsOnTime - 1)
            };
            reg.image_size = new image_size_t
            {
                i_row_wid = (ushort)(layer.InputWidth - 1),
                i_col_high = (ushort)(layer.InputHeight - 1),
                o_row_wid = (ushort)(layer.OutputWidth - 1),
                o_col_high = (ushort)(layer.OutputHeight - 1)
            };
            reg.kernel_pool_type_cfg = new kernel_pool_type_cfg_t
            {
                load_para = 1,
                kernel_type = (byte)layer.KernelType,
                pool_type = (byte)layer.PoolType,
                dma_burst_size = 15,
                pad_value = (byte)layer.PadValue
            };
            reg.kernel_load_cfg = new kernel_load_cfg_t
            {
                load_coor = 1,
                load_time = (byte)(layer.LoadTimes - 1),
                para_size = (uint)layer.OneLoadKernelsSize
            };
            reg.kernel_calc_type_cfg = new kernel_calc_type_cfg_t
            {
                channel_switch_addr = (ushort)(layer.InputRowLength * layer.InputHeight),
                row_switch_addr = (byte)layer.InputRowLength,
                coef_group = (byte)layer.InputGroups,
                load_act = 1
            };
            reg.write_back_cfg = new write_back_cfg_t
            {
                wb_channel_switch_addr = (ushort)(layer.OutputRowLength * layer.OutputHeight),
                wb_row_switch_addr = (byte)layer.OutputRowLength,
                wb_group = (byte)layer.OutputGroups
            };
            reg.conv_value = new conv_value_t
            {
                shr_w = (byte)layer.ShiftW,
                shr_x = (byte)layer.ShiftX,
                arg_w = (uint)layer.ArgW,
                arg_x = (uint)layer.ArgX
            };
            reg.conv_value2 = new conv_value2_t
            {
                arg_add = (ulong)layer.ArgAdd
            };
            reg.dma_parameter = new dma_parameter_t
            {
                channel_byte_num = (ushort)(layer.OutputWidth * layer.OutputHeight - 1),
                dma_total_byte = (uint)(layer.OutputWidth * layer.OutputHeight * layer.OutputChannels - 1)
            };

            bw.Write(reg.interrupt_enabe.Value);
            bw.Write(reg.image_addr.Value);
            bw.Write(reg.image_channel_num.Value);
            bw.Write(reg.image_size.Value);
            bw.Write(reg.kernel_pool_type_cfg.Value);
            bw.Write(reg.kernel_load_cfg.Value);
            bw.Write(reg.kernel_offset.Value);
            bw.Write(reg.kernel_calc_type_cfg.Value);
            bw.Write(reg.write_back_cfg.Value);
            bw.Write(reg.conv_value.Value);
            bw.Write(reg.conv_value2.Value);
            bw.Write(reg.dma_parameter.Value);
        }

        private void DeserializeBinActivation(K210Conv2dLayerArgument argument, K210BinDeserializeContext context)
        {
            var config = argument.Config;
            var sr = context.GetReaderAt((int)argument.ParamAddress.Activation);
            var act = config.ActConfigs = new K210LayerActConfig[16];
            for (int i = 0; i < act.Length; i++)
            {
                var reg = sr.Read<activate_para_t>();
                act[i] = new K210LayerActConfig
                {
                    StartX = ToSigned(reg.x_start, 36),
                    Mul = ToSigned(reg.y_mul, 16),
                    Shift = reg.shift_number
                };
            }

            var bias0 = sr.Read<activate_para_bias0_t>();
            var bias1 = sr.Read<activate_para_bias1_t>();
            for (int i = 0; i < act.Length; i++)
            {
                unsafe
                {
                    if (i < 8)
                        act[i].Add = bias0.result_bias[i];
                    else
                        act[i].Add = bias1.result_bias[i - 8];
                }
            }
        }

        private void DeserializeBinBn(K210Conv2dLayerArgument argument, K210BinDeserializeContext context)
        {
            var config = argument.Config;
            var sr = context.GetReaderAt((int)argument.ParamAddress.Bn);
            var bn = config.BNConfigs = new K210LayerBNConfig[config.OutputChannels];
            for (int i = 0; i < bn.Length; i++)
            {
                var reg = sr.Read<kpu_batchnorm_argument_t>();
                bn[i] = new K210LayerBNConfig
                {
                    Add = ToSigned(reg.norm_add, 32),
                    Mul = ToSigned(reg.norm_mul, 24),
                    Shift = reg.norm_shift
                };
            }
        }

        private void DeserializeBinWeights(K210Conv2dLayerArgument argument, K210BinDeserializeContext context)
        {
            var config = argument.Config;
            var kernelSize = config.KernelType == 0 ? 1 : 3;

            ushort[] weights;
            if (config.IsDepthwise)
                weights = new ushort[config.InputChannels * kernelSize * kernelSize];
            else
                weights = new ushort[config.InputChannels * config.OutputChannels * kernelSize * kernelSize];

            var sr = context.GetReaderAt((int)argument.ParamAddress.Weights);
            if (context.WeightsBits == 8)
            {
                for (int i = 0; i < weights.Length; i++)
                    weights[i] = sr.Read<byte>();
            }
            else
            {
                for (int i = 0; i < weights.Length; i++)
                    weights[i] = sr.Read<ushort>();
            }

            config.Weights = weights;
        }

        private void DeserializeBinLayer(K210Conv2dLayerArgument argument, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt((int)argument.ParamAddress.Layer);
            var reg = sr.Read<kpu_layer_argument_t>();
            var config = argument.Config = new K210ConvLayerConfig();

            config.IsDepthwise = reg.interrupt_enabe.depth_wise_layer == 1;
            config.InputAddress = reg.image_addr.image_src_addr;
            config.OutputAddress = reg.image_addr.image_dst_addr;
            config.InputChannels = reg.image_channel_num.i_ch_num + 1;
            config.OutputChannels = reg.image_channel_num.o_ch_num + 1;
            config.InputWidth = reg.image_size.i_row_wid + 1;
            config.InputHeight = reg.image_size.i_col_high + 1;
            config.OutputWidth = reg.image_size.o_row_wid + 1;
            config.OutputHeight = reg.image_size.o_col_high + 1;
            config.KernelType = reg.kernel_pool_type_cfg.kernel_type;
            config.PoolType = reg.kernel_pool_type_cfg.pool_type;
            config.PadValue = reg.kernel_pool_type_cfg.pad_value;
            config.InputRowLength = reg.kernel_calc_type_cfg.row_switch_addr;
            config.InputGroups = reg.kernel_calc_type_cfg.coef_group;
            config.ArgW = ToSigned(reg.conv_value.arg_w, 24);
            config.ArgX = ToSigned(reg.conv_value.arg_x, 24);
            config.ShiftW = reg.conv_value.shr_w;
            config.ShiftX = reg.conv_value.shr_x;
            config.ArgAdd = ToSigned(reg.conv_value2.arg_add, 40);
        }

        public void Forward(K210Conv2dLayerArgument argument, ForwardContext context)
        {
            var config = argument.Config;

            var src = new byte[config.InputWidth * config.InputHeight * config.InputChannels];
            K210Helper.KpuDownload(context.GetKpuRamAt((int)config.InputAddress), src, config.InputWidth, config.InputHeight, config.InputChannels);

            var convOut = ForwardConvolution(src, config);
            var bnOut = ForwardBatchNorm(convOut, config);
            var actOut = ForwardActivation(bnOut, config);
            var poolOut = config.PoolType == 0 ? actOut : ForwardPooling(actOut, config);

            K210Helper.KpuUpload(context.GetKpuRamAt((int)config.OutputAddress), poolOut, config.OutputWidth, config.OutputHeight, config.OutputChannels);

            if (argument.Flags.HasFlag(K210LayerFlags.MainMemoryOutput))
            {
                K210Helper.KpuDownload(context.GetKpuRamAt((int)config.OutputAddress), context.GetMainRamAt((int)argument.MainMemoryOutputAddress), config.OutputWidth, config.OutputHeight, config.OutputChannels);
            }
        }

        private long[] ForwardConvolution(byte[] src, K210ConvLayerConfig config)
        {
            var weights = config.Weights;
            var argX = config.ArgX;
            var shiftX = config.ShiftX;
            var argW = config.ArgW;
            var shiftW = config.ShiftW;
            var argAdd = config.ArgAdd;
            var imageSize = config.InputWidth * config.InputHeight;
            var padValue = config.PadValue;

            var workspace = new long[config.InputWidth * config.InputHeight * config.OutputChannels];

            if (config.KernelType == 0)
            {
                if (config.IsDepthwise)
                {
                    for (int i = 0; i < workspace.Length; i++)
                    {
                        var x = src[i];
                        var w = weights[i];
                        workspace[i] = x * w + (argX * x >> shiftX) + (argW * w >> shiftW) + argAdd;
                    }
                }
                else
                {
                    for (int oc = 0; oc < config.OutputChannels; oc++)
                    {
                        var localWeights = new ReadOnlySpan<ushort>(weights, oc * config.InputChannels, config.InputChannels);
                        var sumW = localWeights.Sum();

                        for (int i = 0; i < imageSize; i++)
                        {
                            long value = 0;
                            long sumX = 0;
                            for (int ic = 0; ic < config.InputChannels; ic++)
                            {
                                var x = src[ic * imageSize + i];
                                sumX += x;
                                value += localWeights[ic] * x;
                            }

                            workspace[oc * imageSize + i] = value + (argX * sumX >> shiftX) + (argW * sumW >> shiftW) + argAdd * config.InputChannels;
                        }
                    }
                }
            }
            else
            {
                if (config.IsDepthwise)
                {
                    for (int oc = 0; oc < config.OutputChannels; oc++)
                    {
                        var localWeights = new ReadOnlySpan<ushort>(weights, oc * 9, 9);
                        var sumW = localWeights.Sum();

                        for (int oy = 0; oy < config.InputHeight; oy++)
                        {
                            for (int ox = 0; ox < config.InputWidth; ox++)
                            {
                                long value = 0;
                                long sumX = 0;

                                for (int wy = 0; wy < 3; wy++)
                                {
                                    for (int wx = 0; wx < 3; wx++)
                                    {
                                        var iy = oy + wy - 1;
                                        var ix = ox + wx - 1;

                                        var x = iy < 0 || iy >= config.InputHeight || ix < 0 || ix >= config.InputWidth
                                            ? padValue
                                            : src[(oc * config.InputHeight + iy) * config.InputWidth + ix];
                                        var w = localWeights[wy * 3 + wx];
                                        sumX += x;
                                        value += w * x;
                                    }
                                }

                                workspace[(oc * config.InputHeight + oy) * config.InputWidth + ox] = value + (argX * sumX >> shiftX) + (argW * sumW >> shiftW) + argAdd;
                            }
                        }
                    }
                }
                else
                {
                    for (int oc = 0; oc < config.OutputChannels; oc++)
                    {
                        var localWeights = new ReadOnlySpan<ushort>(weights, oc * config.InputChannels * 9, config.InputChannels * 9);
                        var sumW = localWeights.Sum();

                        for (int oy = 0; oy < config.InputHeight; oy++)
                        {
                            for (int ox = 0; ox < config.InputWidth; ox++)
                            {
                                long value = 0;
                                long sumX = 0;

                                for (int ic = 0; ic < config.InputChannels; ic++)
                                {
                                    for (int wy = 0; wy < 3; wy++)
                                    {
                                        for (int wx = 0; wx < 3; wx++)
                                        {
                                            var iy = oy + wy - 1;
                                            var ix = ox + wx - 1;

                                            var x = iy < 0 || iy >= config.InputHeight || ix < 0 || ix >= config.InputWidth
                                                ? padValue
                                                : src[(ic * config.InputHeight + iy) * config.InputWidth + ix];
                                            var w = localWeights[(ic * 3 + wy) * 3 + wx];
                                            sumX += x;
                                            value += w * x;
                                        }
                                    }
                                }

                                workspace[(oc * config.InputHeight + oy) * config.InputWidth + ox] = value + (argX * sumX >> shiftX) + (argW * sumW >> shiftW) + argAdd * config.InputChannels;
                            }
                        }
                    }
                }
            }

            return workspace;
        }

        private long[] ForwardBatchNorm(long[] src, K210ConvLayerConfig config)
        {
            var imageSize = config.InputWidth * config.InputHeight;
            var workspace = new long[config.InputWidth * config.InputHeight * config.OutputChannels];

            for (int oc = 0; oc < config.OutputChannels; oc++)
            {
                var bn = config.BNConfigs[oc];

                for (int i = 0; i < imageSize; i++)
                {
                    var x = src[oc * imageSize + i];
                    workspace[oc * imageSize + i] = (x * bn.Mul >> bn.Shift) + bn.Add;
                }
            }

            return workspace;
        }

        private byte[] ForwardActivation(long[] src, K210ConvLayerConfig config)
        {
            var workspace = new byte[src.Length];

            for (int i = 0; i < src.Length; i++)
            {
                var x = src[i];
                var apply = config.ActConfigs.Last(o => o.StartX <= x);
                workspace[i] = (byte)FxExtensions.Clamp(CarryShift((x - apply.StartX) * apply.Mul, apply.Shift) + apply.Add, 0, 255);
            }

            return workspace;
        }

        private byte[] ForwardPooling(byte[] src, K210ConvLayerConfig config)
        {
            var imageSize = config.InputWidth * config.InputHeight;
            var workspace = new byte[config.OutputWidth * config.OutputHeight * config.OutputChannels];
            int stride = 0, kernelSize = 0;
            string poolType = null;
            switch (config.PoolType)
            {
                case 1:
                    stride = 2;
                    kernelSize = 2;
                    poolType = "max";
                    break;
                case 2:
                    stride = 2;
                    kernelSize = 2;
                    poolType = "ave";
                    break;
                case 3:
                    stride = 4;
                    kernelSize = 4;
                    poolType = "max";
                    break;
                case 4:
                    stride = 4;
                    kernelSize = 4;
                    poolType = "ave";
                    break;
                case 5:
                    stride = 2;
                    kernelSize = 2;
                    poolType = "lt";
                    break;
                default:
                    throw new NotImplementedException();
            }

            var pad = 0;

            for (int oc = 0; oc < config.OutputChannels; oc++)
            {
                var channelSrc = new ReadOnlySpan<byte>(src, imageSize * oc, imageSize);
                for (int oy = 0; oy < config.OutputHeight; oy++)
                {
                    for (int ox = 0; ox < config.OutputWidth; ox++)
                    {
                        int in_x_origin = (int)(ox * stride) - pad;
                        int in_y_origin = (int)(oy * stride) - pad;
                        int kernel_x_start = Math.Max(0, -in_x_origin);
                        int kernel_x_end = Math.Min(kernelSize, config.InputWidth - in_x_origin);
                        int kernel_y_start = Math.Max(0, -in_y_origin);
                        int kernel_y_end = Math.Min(kernelSize, config.InputHeight - in_y_origin);
                        var values = new List<byte>();

                        int kernel_y, kernel_x;
                        for (kernel_y = kernel_y_start; kernel_y < kernel_y_end; kernel_y++)
                        {
                            for (kernel_x = kernel_x_start; kernel_x < kernel_x_end; kernel_x++)
                            {
                                int in_x = in_x_origin + kernel_x;
                                int in_y = in_y_origin + kernel_y;
                                values.Add(channelSrc[in_y * config.InputWidth + in_x]);
                            }
                        }

                        byte value = 0;

                        switch (poolType)
                        {
                            case "max":
                                value = values.Max();
                                break;
                            case "ave":
                                value = (byte)(values.Sum(x => x) / values.Count);
                                break;
                            case "lt":
                                value = values[0];
                                break;
                        }

                        workspace[(oc * config.OutputHeight + oy) * config.OutputWidth + ox] = value;
                    }
                }
            }

            return workspace;
        }

        private static int ToSigned(uint value, int bits)
        {
            var mask = 1U << (bits - 1);
            if (bits != 32 && (value & mask) != 0)
            {
                var sign = 0xFFFFFFFF << bits;
                return (int)(value | sign);
            }

            return (int)value;
        }

        private static long ToSigned(ulong value, int bits)
        {
            var mask = 1UL << (bits - 1);
            if ((value & mask) != 0)
            {
                var sign = 0xFFFFFFFF_FFFFFFFF << bits;
                return (long)(value | sign);
            }

            return (long)value;
        }

        private static long CarryShift(long value, int shift)
        {
            if (shift > 0)
            {
                value >>= shift - 1;
                if ((value & 0x1) != 0)
                {
                    if (value < 0)
                        value = (value >> 1) - 1;
                    else
                        value = (value >> 1) + 1;
                }
                else
                {
                    value >>= 1;
                }
            }

            return value;
        }
    }
}
