// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using TorchSharp;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using SMath = System.Math;

namespace Nncase.Utilities;

/// <summary>
/// Array utility.
/// </summary>
public static class QuantUtility
{
    public enum AdaMode
    {
        /// <summary>
        /// Conv2D.
        /// </summary>
        Conv2D,

        /// <summary>
        /// Conv2DTranspose.
        /// </summary>
        Conv2DTranspose,

        /// <summary>
        /// Linear.
        /// </summary>
        Linear,
    }

    /// <summary>
    /// GetQuantParam.
    /// </summary>
    public static QuantParam GetQuantParam(ValueRange<float> range, int bits, QuantMode quantMode)
    {
        range = FixupRange(range, quantMode == QuantMode.SignedSymmetricMode);
        double qMax;
        double qMin;
        switch (quantMode)
        {
            case QuantMode.UnsignedMode:
                qMin = 0;
                qMax = (1 << bits) - 1;
                break;
            case QuantMode.SignedSymmetricMode:
                qMin = -(1 << (bits - 1)) + 1;
                qMax = (1 << (bits - 1)) - 1;
                break;
            case QuantMode.SignedAsymmetricMode:
                qMin = -(1 << (bits - 1));
                qMax = (1 << (bits - 1)) - 1;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(quantMode), "Invalid QuantMode");
        }

        var scale = (range.Max - range.Min) / (qMax - qMin);
        var bias = SMath.Round(range.Min * (qMin - qMax) / (range.Max - range.Min)) + qMin;
        return new QuantParam((int)bias, (float)scale);
    }

    /// <summary>
    /// fixup range.
    /// </summary>
    public static ValueRange<float> FixupRange(ValueRange<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            var r = SMath.Max(SMath.Max(SMath.Abs(range.Min), SMath.Abs(range.Max)), 0.01f);
            return (-r, r);
        }
        else
        {
            range.Max = SMath.Max(0, range.Max);
            range.Min = SMath.Min(0, range.Min);
            var r = range.Max - range.Min;
            if (r == 0)
            {
                r = 0.1f;
            }
            else if (r < 0.01f)
            {
                r = 0.01f;
            }

            range.Max = range.Min + r;
        }

        return range;
    }

    public static ValueRange<T> GetRange<T>(Span<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var data = input.ToArray();
        return new(data.Min(), data.Max());
    }

    public static List<float> GetWeightsRangesByChannel(Span<float> weights, int channels)
    {
        var tmpMin = float.MaxValue;
        var tmpMax = float.MinValue;
        var minMaxArr = new List<float>();
        for (int i = 0; i < weights.Length; i++)
        {
            if (i % (weights.Length / channels) == 0)
            {
                tmpMin = float.MaxValue;
                tmpMax = float.MinValue;
            }

            if (weights[i] < tmpMin)
            {
                tmpMin = weights[i];
            }

            if (weights[i] > tmpMax)
            {
                tmpMax = weights[i];
            }

            if ((i + 1) % (weights.Length / channels) == 0)
            {
                tmpMax = Math.Max(0, minMaxArr[i + 1]);
                tmpMin = Math.Min(0, minMaxArr[i]);
                var r = tmpMax - tmpMin;
                if (r == 0)
                {
                    r = 0.1f;
                }
                else if (r < 0.01f)
                {
                    r = 0.01f;
                }

                tmpMax = tmpMin + r;

                minMaxArr.Add(tmpMin);
                minMaxArr.Add(tmpMax);
            }
        }

        return minMaxArr;
    }

    public static Span<float> SquantWeights(Span<float> inputWeights, Expr inputWeightsRanges, Nncase.IR.Shape inputWeightsShape, QuantMode quantMode, int bits, bool isByChannel)
    {
        // isByChannel = false;
        // int O = 1000; //32;
        // int C = 1280; //3;
        // int R = 1; //3;
        // int S = 1; //3;
        // float []  n = new float[O*C*R*S];
        // float []  n_gt = new float[O*C*R*S];
        // float [,]  n_range = new float[O,2];
        // BinaryReader br = new BinaryReader(new FileStream("/data/jinxiaocheng/x.bin", FileMode.Open));
        // //BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/x_dequant.bin", FileMode.Open));
        // //BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/quant_tensor.bin", FileMode.Open));
        // //BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/delta.bin", FileMode.Open));
        // //BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/zero_point.bin", FileMode.Open));
        // BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/x_int.bin", FileMode.Open));
        // //BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/x_round.bin", FileMode.Open));
        // BinaryReader br_min = new BinaryReader(new FileStream("/data/jinxiaocheng/x_min.bin", FileMode.Open));
        // BinaryReader br_max = new BinaryReader(new FileStream("/data/jinxiaocheng/x_max.bin", FileMode.Open));
        // for (var i = 0; i < (O*C*R*S); i++)
        // {
        //     n[i] = br.ReadSingle();
        //     n_gt[i] = br_gt.ReadSingle();
        // }
        // /*for (var i = 0; i < (32); i++)
        // {
        //     n_gt[27*i] = br_gt.ReadSingle();
        //     for (var j = 1; j < 27; j++)
        //     {
        //         n_gt[27*i + j] = n_gt[27*i];
        //     }
        // }*/
        // for (var i = 0; i < (isByChannel?O:1); i++)
        // {
        //     n_range[i, 0] = br_min.ReadSingle();
        //     n_range[i, 1] = br_max.ReadSingle();
        // }
        // //n_range[0, 0] = -0.24556864798069f;
        // //n_range[0, 1] = 0.3308314383029938f;
        // br.Close();
        // br_gt.Close();
        // br_min.Close();
        // br_max.Close();
        // inputWeights = new Span<float>(n);
        // Span<float> gt = new Span<float>(n_gt);
        // Tensor tmp = n_range;
        // inputWeightsRanges = Const.FromTensor(tmp);
        // // inputWeightsShape = new Nncase.IR.Shape(O,C,R,S);
        // inputWeightsShape = new Nncase.IR.Shape(O,C);
        float qmax, qmin;
        if (quantMode == QuantMode.UnsignedMode)
        {
            qmax = (1 << bits) - 1;
            qmin = 0;
        }
        else if (quantMode == QuantMode.SignedAsymmetricMode)
        {
            qmax = (1 << (bits - 1)) - 1;
            qmin = -(1 << (bits - 1));
        }
        else
        {
            qmax = (1 << (bits - 1)) - 1;
            qmin = -(1 << (bits - 1)) + 1;
        }

        torch.Tensor x, delta, zero_point;
        x = torch.from_array(inputWeights.ToArray());
        if (inputWeightsShape.Rank == 4)
        {
            var out_channel = inputWeightsShape[0];
            var in_channel = inputWeightsShape[1];
            var filter_h = inputWeightsShape[2];
            var filter_w = inputWeightsShape[3];
            x = x.reshape(new long[] { out_channel.FixedValue, in_channel.FixedValue, filter_h.FixedValue, filter_w.FixedValue });
            if (isByChannel)
            {
                delta = torch.ones_like(x);
                zero_point = torch.zeros_like(x);
                for (var c = 0; c < out_channel.FixedValue; c++)
                {
                    var x_min = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[2 * c];
                    var x_max = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[(2 * c) + 1];
                    var delta_tmp = (x_max - x_min) / (qmax - qmin);
                    var zero_point_tmp = System.Math.Round(((x_max * qmin) - (x_min * qmax)) / (x_max - x_min));
                    delta[c] = torch.full_like(x[c], delta_tmp);
                    zero_point[c] = torch.full_like(x[c], zero_point_tmp);
                }
            }
            else
            {
                throw new NotSupportedException("By layer weights quant is not supported.");
            }
        }
        else
        {
            var out_channel = inputWeightsShape[0];
            var in_channel = inputWeightsShape[1];
            x = x.reshape(new long[] { out_channel.FixedValue, in_channel.FixedValue });
            if (isByChannel)
            {
                delta = torch.ones_like(x);
                zero_point = torch.zeros_like(x);
                for (var c = 0; c < out_channel.FixedValue; c++)
                {
                    var x_min = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[2 * c];
                    var x_max = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[(2 * c) + 1];
                    var delta_tmp = (x_max - x_min) / (qmax - qmin);
                    var zero_point_tmp = System.Math.Round(((x_max * qmin) - (x_min * qmax)) / (x_max - x_min));
                    delta[c] = torch.full_like(x[c], delta_tmp);
                    zero_point[c] = torch.full_like(x[c], zero_point_tmp);
                }
            }
            else
            {
                throw new NotSupportedException("By layer weights quant is not supported.");
            }
        }

        var quant_tensor = (x / delta) + zero_point;
        var x_int = Adaptive_round(quant_tensor, qmin, qmax); // SQuant量化
        var x_quant = torch.clamp(x_int, torch.tensor(qmin), torch.tensor(qmax));
        var x_dequant = (x_quant - zero_point) * delta;

        // var x_dequant = x_int; //x_int; //zero_point; //delta; //quant_tensor; //(x_quant - zero_point) * delta;
        var rst = new Span<float>(x_dequant.data<float>().ToArray());

        // for (var i = 0; i < (O*C*R*S); i++)
        // {
        //     if ((rst[i] - gt[i]) != 0)
        //     {
        //         System.Console.WriteLine("{0} {1} {2}", rst[i], gt[i], (rst[i] - gt[i]) / gt[i]);
        //     }
        // }
        // System.Console.WriteLine(inputWeights.ToArray()[0]);
        // System.Console.WriteLine(rst.ToArray()[0]);
        return rst;

        // inputWeightsRanges is pre calculated by range optimization, so when compute inputWeights quant parameters, range should be gotten from inputWeightsRanges, but not
        // be gotten from inputWeights here simply. And for quantMode, there are 3 modes, UnsignedMode is easy to understand, and for SignedAsymmetricMode/SignedSymmetricMode,
        // it effects Qmax/Qmin for quant function, for example, k510 int8 needs SignedAsymmetricMode, and k230 int8/int16 needs SignedSymmetricMode, please refer GetQuantParam() in this file.

        // System.Console.WriteLine(((Tensor<float>)(((TensorConst)(inputWeightsRanges)).Value)).ToArray()[0]);
        // System.Console.WriteLine(inputWeights.ToArray()[0]);
        // return inputWeights;
    }

    public static Span<float> AdaRoundWeights(Span<float> inputWeights, Expr inputWeightsRanges, Nncase.IR.Shape inputWeightsShape, List<Tensor> layerInput, List<Tensor> layerOutputGT, QuantMode quantMode, int bits, bool isByChannel, Expr paddings, Expr strides, Expr dilations, Expr groups, float startB, float endB, int iters, int deviceID, float warmup, float weightParam, AdaMode adamode)
    {
        // adamode = AdaMode.Linear;
        // bits = 4;
        // isByChannel = true;
        // int [,] pad_buf = new int[,] {{1, 1}, {1, 1}};
        // Tensor tmp = pad_buf;
        // paddings = Const.FromTensor(tmp);
        // int [] strides_buf = new int[] {2, 2};
        // tmp = strides_buf;
        // strides = Const.FromTensor(tmp);
        // int [] dilations_buf = new int[] {1, 1};
        // tmp = dilations_buf;
        // dilations = Const.FromTensor(tmp);
        // int [] groups_buf = new int[] {1};
        // tmp = groups_buf;
        // groups = Const.FromTensor(tmp);
        // startB = 20;
        // endB = 2;
        // iters = 10;
        // warmup = 0.2f;
        // weightParam = 0.1f;
        // int O = 32; //1000; //32;
        // int C = 3; //1280; //3;
        // int R = 3; //1; //3;
        // int S = 3; //1; //3;
        // int N = 2;
        // int H_in0 = 224; //1; //224;
        // int W_in0 = 224; //1; //224;
        // int H_out0 = 112; //1; //112;
        // int W_out0 = 112; //1; //112;
        // float []  cached_inps0_buf = new float[N*C*H_in0*W_in0];
        // float []  cached_inps1_buf = new float[N*C*H_in0*W_in0];
        // float []  cached_inps2_buf = new float[N*C*H_in0*W_in0];
        // float []  cached_outs0_buf = new float[N*O*H_out0*W_out0];
        // float []  cached_outs1_buf = new float[N*O*H_out0*W_out0];
        // float []  cached_outs2_buf = new float[N*O*H_out0*W_out0];
        // float []  x_buf = new float[O*C*R*S];
        // float []  gt_buf = new float[O*C*R*S];
        // float [,]  range_buf = new float[O,2];
        // BinaryReader br_cached_inps0 = new BinaryReader(new FileStream("/data/jinxiaocheng/cached_inps0.bin", FileMode.Open));
        // BinaryReader br_cached_inps1 = new BinaryReader(new FileStream("/data/jinxiaocheng/cached_inps1.bin", FileMode.Open));
        // BinaryReader br_cached_inps2 = new BinaryReader(new FileStream("/data/jinxiaocheng/cached_inps2.bin", FileMode.Open));
        // BinaryReader br_cached_outs0 = new BinaryReader(new FileStream("/data/jinxiaocheng/cached_outs0.bin", FileMode.Open));
        // BinaryReader br_cached_outs1 = new BinaryReader(new FileStream("/data/jinxiaocheng/cached_outs1.bin", FileMode.Open));
        // BinaryReader br_cached_outs2 = new BinaryReader(new FileStream("/data/jinxiaocheng/cached_outs2.bin", FileMode.Open));
        // BinaryReader br_x = new BinaryReader(new FileStream("/data/jinxiaocheng/x.bin", FileMode.Open));
        // BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/alpha.bin", FileMode.Open));
        // // BinaryReader br_gt = new BinaryReader(new FileStream("/data/jinxiaocheng/alpha_int.bin", FileMode.Open));
        // BinaryReader br_min = new BinaryReader(new FileStream("/data/jinxiaocheng/x_min.bin", FileMode.Open));
        // BinaryReader br_max = new BinaryReader(new FileStream("/data/jinxiaocheng/x_max.bin", FileMode.Open));
        // for (var i = 0; i < (N*C*H_in0*W_in0); i++)
        // {
        //     cached_inps0_buf[i] = br_cached_inps0.ReadSingle();
        //     cached_inps1_buf[i] = br_cached_inps1.ReadSingle();
        //     cached_inps2_buf[i] = br_cached_inps2.ReadSingle();
        // }
        // br_cached_inps0.Close();
        // br_cached_inps1.Close();
        // br_cached_inps2.Close();
        // for (var i = 0; i < (N*O*H_out0*W_out0); i++)
        // {
        //     cached_outs0_buf[i] = br_cached_outs0.ReadSingle();
        //     cached_outs1_buf[i] = br_cached_outs1.ReadSingle();
        //     cached_outs2_buf[i] = br_cached_outs2.ReadSingle();
        // }
        // br_cached_outs0.Close();
        // br_cached_outs1.Close();
        // br_cached_outs2.Close();
        // for (var i = 0; i < (O*C*R*S); i++)
        // {
        //     x_buf[i] = br_x.ReadSingle();
        //     gt_buf[i] = br_gt.ReadSingle();
        // }
        // br_x.Close();
        // br_gt.Close();
        // for (var i = 0; i < (isByChannel?O:1); i++)
        // {
        //     range_buf[i, 0] = br_min.ReadSingle();
        //     range_buf[i, 1] = br_max.ReadSingle();
        // }
        // br_min.Close();
        // br_max.Close();
        // inputWeights = new Span<float>(x_buf);
        // tmp = range_buf;
        // inputWeightsRanges = Const.FromTensor(tmp);
        // if (adamode == AdaMode.Linear)
        // {
        //     inputWeightsShape = new Nncase.IR.Shape(O,C);
        //     layerInput.Clear();
        //     tmp = new Tensor<float>(cached_inps0_buf, new[] { N,C });
        //     layerInput.Add(tmp);
        //     tmp = new Tensor<float>(cached_inps1_buf, new[] { N,C });
        //     layerInput.Add(tmp);
        //     tmp = new Tensor<float>(cached_inps2_buf, new[] { N,C });
        //     layerInput.Add(tmp);
        //     layerOutputGT.Clear();
        //     tmp = new Tensor<float>(cached_outs0_buf, new[] { N,O });
        //     layerOutputGT.Add(tmp);
        //     tmp = new Tensor<float>(cached_outs1_buf, new[] { N,O });
        //     layerOutputGT.Add(tmp);
        //     tmp = new Tensor<float>(cached_outs2_buf, new[] { N,O });
        //     layerOutputGT.Add(tmp);
        // }
        // else
        // {
        //     inputWeightsShape = new Nncase.IR.Shape(O,C,R,S);
        //     layerInput.Clear();
        //     tmp = new Tensor<float>(cached_inps0_buf, new[] { N,C,H_in0,W_in0 });
        //     layerInput.Add(tmp);
        //     tmp = new Tensor<float>(cached_inps1_buf, new[] { N,C,H_in0,W_in0 });
        //     layerInput.Add(tmp);
        //     tmp = new Tensor<float>(cached_inps2_buf, new[] { N,C,H_in0,W_in0 });
        //     layerInput.Add(tmp);
        //     layerOutputGT.Clear();
        //     tmp = new Tensor<float>(cached_outs0_buf, new[] { N,O,H_out0,W_out0 });
        //     layerOutputGT.Add(tmp);
        //     tmp = new Tensor<float>(cached_outs1_buf, new[] { N,O,H_out0,W_out0 });
        //     layerOutputGT.Add(tmp);
        //     tmp = new Tensor<float>(cached_outs2_buf, new[] { N,O,H_out0,W_out0 });
        //     layerOutputGT.Add(tmp);
        // }
        int n = layerInput.Count;
        torch.Tensor cur_inp, cur_out;
        int stride_h = 0, stride_w = 0, padding_h_s = 0, padding_w_s = 0, padding_h_e = 0, padding_w_e = 0, dilation_h = 0, dilation_w = 0, group = 0;
        int start_decay = (int)(warmup * (float)iters);
        if (adamode == AdaMode.Conv2D || adamode == AdaMode.Conv2DTranspose)
        {
            stride_h = ((Tensor)((TensorConst)strides).Value).ToArray<int>()[0];
            stride_w = ((Tensor)((TensorConst)strides).Value).ToArray<int>()[1];
            padding_h_s = ((Tensor)((TensorConst)paddings).Value).ToArray<int>()[0];
            padding_h_e = ((Tensor)((TensorConst)paddings).Value).ToArray<int>()[1];
            padding_w_s = ((Tensor)((TensorConst)paddings).Value).ToArray<int>()[2];
            padding_w_e = ((Tensor)((TensorConst)paddings).Value).ToArray<int>()[3];
            dilation_h = ((Tensor)((TensorConst)dilations).Value).ToArray<int>()[0];
            dilation_w = ((Tensor)((TensorConst)dilations).Value).ToArray<int>()[1];
            group = ((Tensor)((TensorConst)groups).Value).ToArray<int>()[0];
        }

        float qmax, qmin;
        if (quantMode == QuantMode.UnsignedMode)
        {
            qmax = (1 << bits) - 1;
            qmin = 0;
        }
        else if (quantMode == QuantMode.SignedAsymmetricMode)
        {
            qmax = (1 << (bits - 1)) - 1;
            qmin = -(1 << (bits - 1));
        }
        else
        {
            qmax = (1 << (bits - 1)) - 1;
            qmin = -(1 << (bits - 1)) + 1;
        }

        torch.Tensor x, delta, zero_point;
        x = torch.from_array(inputWeights.ToArray());
        if (inputWeightsShape.Rank == 4)
        {
            var out_channel = inputWeightsShape[0];
            var in_channel = inputWeightsShape[1];
            var filter_h = inputWeightsShape[2];
            var filter_w = inputWeightsShape[3];
            x = x.reshape(new long[] { out_channel.FixedValue, in_channel.FixedValue, filter_h.FixedValue, filter_w.FixedValue });
            if (isByChannel)
            {
                delta = torch.ones_like(x);
                zero_point = torch.zeros_like(x);
                for (var c = 0; c < out_channel.FixedValue; c++)
                {
                    var x_min = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[2 * c];
                    var x_max = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[(2 * c) + 1];
                    var delta_tmp = (x_max - x_min) / (qmax - qmin);
                    var zero_point_tmp = System.Math.Round(((x_max * qmin) - (x_min * qmax)) / (x_max - x_min));
                    delta[c] = torch.full_like(x[c], delta_tmp);
                    zero_point[c] = torch.full_like(x[c], zero_point_tmp);
                }
            }
            else
            {
                throw new NotSupportedException("By layer weights quant is not supported.");
            }
        }
        else
        {
            var out_channel = inputWeightsShape[0];
            var in_channel = inputWeightsShape[1];
            x = x.reshape(new long[] { out_channel.FixedValue, in_channel.FixedValue });
            if (isByChannel)
            {
                delta = torch.ones_like(x);
                zero_point = torch.zeros_like(x);
                for (var c = 0; c < out_channel.FixedValue; c++)
                {
                    var x_min = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[2 * c];
                    var x_max = ((Tensor<float>)((TensorConst)inputWeightsRanges).Value).ToArray()[(2 * c) + 1];
                    var delta_tmp = (x_max - x_min) / (qmax - qmin);
                    var zero_point_tmp = System.Math.Round(((x_max * qmin) - (x_min * qmax)) / (x_max - x_min));
                    delta[c] = torch.full_like(x[c], delta_tmp);
                    zero_point[c] = torch.full_like(x[c], zero_point_tmp);
                }
            }
            else
            {
                throw new NotSupportedException("By layer weights quant is not supported.");
            }
        }

        const float gamma = -0.1f;
        const float zeta = 1.1f;
        var x_floor = (x / delta).floor();
        var rest = (x / delta) - x_floor;
        var alpha = -torch.log(((zeta - gamma) / (rest - gamma)) - 1); // 待学习的张量alpha，初始值
        var alpha_para = TorchSharp.torch.nn.Parameter(alpha);
        var alpha_para_init = alpha_para.clone();
        IEnumerable<TorchSharp.Modules.Parameter> parameters = new List<TorchSharp.Modules.Parameter>() { alpha_para };
        var optimizer = torch.optim.Adam(parameters, 1e-3); // 优化器

        torch.Tensor qmin_tensor = qmin;
        torch.Tensor qmax_tensor = qmax;

        // 训练迭代
        for (int i = 0; i < iters; i++)
        {
            int idx = i % n; // 原来是随机

            if (adamode == AdaMode.Linear)
            {
                var n_in = layerInput[idx].Shape[0];
                var c_in = layerInput[idx].Shape[1];
                cur_inp = torch.from_array(layerInput[idx].ToArray<float>());
                cur_inp = cur_inp.reshape(new long[] { n_in.FixedValue, c_in.FixedValue });

                var n_out = layerOutputGT[idx].Shape[0];
                var c_out = layerOutputGT[idx].Shape[1];
                cur_out = torch.from_array(layerOutputGT[idx].ToArray<float>());
                cur_out = cur_out.reshape(new long[] { n_out.FixedValue, c_out.FixedValue });
            }
            else
            {
                var n_in = layerInput[idx].Shape[0];
                var c_in = layerInput[idx].Shape[1];
                var h_in = layerInput[idx].Shape[2];
                var w_in = layerInput[idx].Shape[3];
                cur_inp = torch.from_array(layerInput[idx].ToArray<float>());
                cur_inp = cur_inp.reshape(new long[] { n_in.FixedValue, c_in.FixedValue, h_in.FixedValue, w_in.FixedValue });

                var n_out = layerOutputGT[idx].Shape[0];
                var c_out = layerOutputGT[idx].Shape[1];
                var h_out = layerOutputGT[idx].Shape[2];
                var w_out = layerOutputGT[idx].Shape[3];
                cur_out = torch.from_array(layerOutputGT[idx].ToArray<float>());
                cur_out = cur_out.reshape(new long[] { n_out.FixedValue, c_out.FixedValue, h_out.FixedValue, w_out.FixedValue });
            }

            // 软量化权重
            var x_int = x_floor + torch.clamp((torch.sigmoid(alpha_para) * (zeta - gamma)) + gamma, 0, 1);
            var x_quant = torch.clamp(x_int + zero_point, qmin_tensor, qmax_tensor);
            var x_float_q = (x_quant - zero_point) * delta;

            // reset gradient，清空过往梯度，为下一波梯度累加做准备
            optimizer.zero_grad();

            // conv2 or linear forward
            torch.Tensor out_quant;
            if (adamode == AdaMode.Conv2D)
            {
                // torch不支持非对称padding conv，需要拆算子
                if (padding_h_s != padding_h_e || padding_w_s != padding_w_e)
                {
                    var padding_tmp = torch.nn.functional.pad(cur_inp, new long[] { padding_w_s, padding_w_e, padding_h_s, padding_h_e });
                    out_quant = torch.nn.functional.conv2d(
                        padding_tmp,
                        x_float_q,
                        strides: new long[] { stride_h, stride_w },
                        padding: new long[] { 0, 0 },
                        dilation: new long[] { dilation_h, dilation_w },
                        groups: group);
                }
                else
                {
                    out_quant = torch.nn.functional.conv2d(
                        cur_inp,
                        x_float_q,
                        strides: new long[] { stride_h, stride_w },
                        padding: new long[] { padding_h_s, padding_w_s },
                        dilation: new long[] { dilation_h, dilation_w },
                        groups: group);
                }
            }
            else if (adamode == AdaMode.Conv2DTranspose)
            {
                // torch不支持非对称padding conv_transpose2d，需要拆算子
                if (padding_h_s != padding_h_e || padding_w_s != padding_w_e)
                {
                    var out_tmp = torch.nn.functional.conv_transpose2d(
                        cur_inp,
                        x_float_q,
                        strides: new long[] { stride_h, stride_w },
                        padding: new long[] { 0, 0 },
                        dilation: new long[] { dilation_h, dilation_w },
                        groups: group);
                    out_quant = out_tmp.index(new TensorIndex[]
                    {
                        TensorIndex.Slice(0, layerInput[idx].Shape[0].FixedValue),
                        TensorIndex.Slice(0, layerInput[idx].Shape[1].FixedValue),
                        TensorIndex.Slice(padding_h_s, layerInput[idx].Shape[2].FixedValue - padding_h_e),
                        TensorIndex.Slice(padding_w_s, layerInput[idx].Shape[3].FixedValue - padding_w_e),
                    });
                }
                else
                {
                    out_quant = torch.nn.functional.conv_transpose2d(
                        cur_inp,
                        x_float_q,
                        strides: new long[] { stride_h, stride_w },
                        padding: new long[] { padding_h_e, padding_w_e },
                        dilation: new long[] { dilation_h, dilation_w },
                        groups: group);
                }
            }
            else
            {
                out_quant = torch.nn.functional.linear(cur_inp, x_float_q);
            }

            // loss
            var rec_loss = (out_quant - cur_out).abs().pow(2.0).sum(1).mean();
            torch.Tensor round_loss;
            float b;
            if ((i + 1) < start_decay)
            {
                b = 0.0f;
                round_loss = torch.zeros_like(rec_loss);
            }
            else
            {
                b = (float)endB + (((float)startB - (float)endB) * SMath.Max(0.0f, 1.0f - ((float)(i + 1 - start_decay) / (float)(iters - start_decay))));
                var round_vals = torch.clamp((torch.sigmoid(alpha_para) * (zeta - gamma)) + gamma, 0, 1);
                round_loss = weightParam * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum();
            }

            var err = rec_loss + round_loss;

            // backward
            err.backward();

            // 梯度下降，更新alpha值
            optimizer.step();
        }

        var x_floor_rst = (x / delta).floor();
        var x_int_rst = x_floor_rst + (alpha_para >= torch.tensor(0.0f));
        var x_quant_rst = torch.clamp(x_int_rst + zero_point, qmin_tensor, qmax_tensor);
        var x_float_q_rst = (x_quant_rst - zero_point) * delta;

        // x.print();
        // x_float_q_rst.print();
        // (alpha_para_init - alpha_para).print();
        // (x_int_rst - x_floor_rst).print();

        // Span<float> rst = new Span<float>(alpha_para.data<float>().ToArray());
        // for (var i = 0; i < (O*C*R*S); i++)
        // {
        //     if ((rst[i] - gt_buf[i]) != 0)
        //     {
        //         System.Console.WriteLine("{0} {1} {2}", rst[i], gt_buf[i], (rst[i] - gt_buf[i]) / gt_buf[i]);
        //     }
        // }
        var rst = new Span<float>(x_float_q_rst.data<float>().ToArray());

        return rst;

        // return inputWeights;
    }

    private static void Rounding_forward(float rounding_error_sum, ref torch.Tensor rounding_number_, ref torch.Tensor rounding_error_, torch.Tensor number_, torch.Tensor error_, ref torch.Tensor priority_, torch.Tensor order_, ref torch.Tensor priority_1)
    {
        int topk = (int)System.Math.Round(System.Math.Abs(rounding_error_sum));
        bool over_squant = topk >= System.Math.Abs(rounding_error_sum);
        if (topk > 0)
        {
            var order_tmp = order_.slice(0, 0, topk, 1);
            rounding_error_.index_put_(error_.index(order_tmp), order_tmp);
            rounding_number_.index_put_(number_.index(order_tmp), order_tmp);
            if (over_squant)
            {
                var idx_c = order_[topk - 1];
                priority_1[idx_c] = rounding_error_[idx_c].abs();
            }
            else
            {
                var idx_c = order_[topk];
                priority_[idx_c] = rounding_error_[idx_c].abs();
            }
        }
    }

    private static void SQuant_func(torch.Tensor rounding_error_sum, ref torch.Tensor rounding_number, ref torch.Tensor rounding_error, torch.Tensor up_number, torch.Tensor up_error, ref torch.Tensor up_priority, torch.Tensor up_order, torch.Tensor down_number, torch.Tensor down_error, ref torch.Tensor down_priority, torch.Tensor down_order)
    {
        var rounding_number_shape = rounding_number.shape;
        var batch_size = rounding_number_shape[0];
        var input_channel = rounding_number_shape[1];
        for (var n = 0; n < batch_size; n++)
        {
            for (var c = 0; c < input_channel; c++)
            {
                if ((float)rounding_error_sum[n][c] < 0)
                {
                    var rounding_number_ = rounding_number[n][c];
                    var rounding_error_ = rounding_error[n][c];
                    var priority_ = up_priority[n][c];
                    var priority_1 = down_priority[n][c];
                    Rounding_forward((float)rounding_error_sum[n][c], ref rounding_number_, ref rounding_error_, up_number[n][c], up_error[n][c], ref priority_, up_order[n][c], ref priority_1);
                    rounding_number[n][c] = rounding_number_;
                    rounding_error[n][c] = rounding_error_;
                    up_priority[n][c] = priority_;
                    down_priority[n][c] = priority_1;
                }
                else
                {
                    var rounding_number_ = rounding_number[n][c];
                    var rounding_error_ = rounding_error[n][c];
                    var priority_ = down_priority[n][c];
                    var priority_1 = up_priority[n][c];
                    Rounding_forward((float)rounding_error_sum[n][c], ref rounding_number_, ref rounding_error_, down_number[n][c], down_error[n][c], ref priority_, down_order[n][c], ref priority_1);
                    rounding_number[n][c] = rounding_number_;
                    rounding_error[n][c] = rounding_error_;
                    down_priority[n][c] = priority_;
                    up_priority[n][c] = priority_1;
                }
            }
        }
    }

    private static torch.Tensor Adaptive_round(torch.Tensor x, float t_min, float t_max)
    {
        bool squant_k = true;
        bool squant_c = true;

        var rounding_number = x.round(); // round取整值
        var rounding_error = rounding_number - x; // 误差
        var zeros = torch.zeros_like(rounding_error);

        var up_number = rounding_number.clone();
        var up_error = rounding_error.clone();
        up_error = torch.where(x >= t_max, zeros, up_error); // 边界上的值不能再调整，所以去除
        up_error = torch.where(up_error > 0, zeros, up_error); // 误差为正的都设为0，即up对应“原值>量化值”的集合
        var up_priority = up_error.clone().abs();

        up_error = torch.where(up_error != 0, up_error + 1, up_error); // up集合中，Flip翻转后对应的误差
        up_number = torch.where(up_error != 0, up_number + 1, up_number); // up集合中，Flip翻转后对应的取整值

        var down_number = rounding_number.clone();
        var down_error = rounding_error.clone();
        down_error = torch.where(x <= t_min, zeros, down_error); // 边界上的值不能再调整，所以去除
        down_error = torch.where(down_error < 0, zeros, down_error); // 误差为负的都设为0，即down对应“原值<量化值”的集合
        var down_priority = down_error.clone().abs();

        down_error = torch.where(down_error != 0, down_error - 1, down_error); // down集合中，Flip翻转后对应的误差
        down_number = torch.where(down_error != 0, down_number - 1, down_number); // down集合中，Flip翻转后对应的取整值

        var x_tmp = x.reshape(new long[] { x.size(0), x.size(1), -1 });
        var conver_shape = x_tmp.shape; // HW维度合并
        if (conver_shape[2] == 1)
        {
            squant_k = false; // 只有一个元素时， 不做K的逼近
        }

        if (squant_k)
        {
            var rounding_error_sum = rounding_error.reshape(conver_shape).sum(-1);
            var sort_ret = torch.sort(up_priority.reshape(conver_shape), -1, true);
            torch.Tensor up_order = sort_ret.Indices;
            sort_ret = torch.sort(down_priority.reshape(conver_shape), -1, true);
            torch.Tensor down_order = sort_ret.Indices;
            up_priority *= 0.0;
            down_priority *= 0.0;

            rounding_number = rounding_number.reshape(conver_shape);
            rounding_error = rounding_error.reshape(conver_shape);
            up_number = up_number.reshape(conver_shape);
            up_error = up_error.reshape(conver_shape);
            up_priority = up_priority.reshape(conver_shape);
            down_number = down_number.reshape(conver_shape);
            down_error = down_error.reshape(conver_shape);
            down_priority = down_priority.reshape(conver_shape);
            SQuant_func(rounding_error_sum, ref rounding_number, ref rounding_error, up_number, up_error, ref up_priority, up_order, down_number, down_error, ref down_priority, down_order);
            rounding_number = rounding_number.reshape(x.shape);
            rounding_error = rounding_error.reshape(x.shape);
            up_priority = up_priority.reshape(x.shape);
            down_priority = down_priority.reshape(x.shape);
        }

        if (squant_c)
        {
            conver_shape = new long[] { 1, x.size(0), -1 };
            var rounding_error_sum = rounding_error.reshape(conver_shape).sum(-1);
            var sort_ret = torch.sort(up_priority.reshape(conver_shape), -1, true);
            var up_order = sort_ret.Indices;
            sort_ret = torch.sort(down_priority.reshape(conver_shape), -1, true);
            var down_order = sort_ret.Indices;

            rounding_number = rounding_number.reshape(conver_shape);
            rounding_error = rounding_error.reshape(conver_shape);
            up_number = up_number.reshape(conver_shape);
            up_error = up_error.reshape(conver_shape);
            up_priority = up_priority.reshape(conver_shape);
            down_number = down_number.reshape(conver_shape);
            down_error = down_error.reshape(conver_shape);
            down_priority = down_priority.reshape(conver_shape);
            SQuant_func(rounding_error_sum, ref rounding_number, ref rounding_error, up_number, up_error, ref up_priority, up_order, down_number, down_error, ref down_priority, down_order);
        }

        rounding_number = rounding_number.reshape(x.shape);
        rounding_error = rounding_error.reshape(x.shape);
        up_priority = up_priority.reshape(x.shape);
        down_priority = down_priority.reshape(x.shape);

        return rounding_number;
    }
}
