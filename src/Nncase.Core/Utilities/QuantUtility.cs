// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SMath = System.Math;
using TorchSharp;

namespace Nncase.Utilities;

/// <summary>
/// Array utility.
/// </summary>
public static class QuantUtility
{
    /// <summary>
    /// GetQuantParam
    /// </summary>
    /// <param name="range"></param>
    /// <param name="bits"></param>
    /// <param name="quantMode"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static QuantParam GetQuantParam(ValueRange<float> range, int bits, QuantMode quantMode)
    {
        range = FixupRange(range, quantMode == QuantMode.SignedSymmetricMode);
        double QMax = 255;
        double QMin = 0;
        switch (quantMode)
        {
            case QuantMode.UnsignedMode:
                QMin = 0;
                QMax = (1 << bits) - 1;
                break;
            case QuantMode.SignedSymmetricMode:
                QMin = -(1 << (bits - 1)) + 1;
                QMax = (1 << (bits - 1)) - 1;
                break;
            case QuantMode.SignedAsymmetricMode:
                QMin = -(1 << (bits - 1));
                QMax = (1 << (bits - 1)) - 1;
                break;
            default:
                throw new ArgumentOutOfRangeException("Invalid QuantMode");
        }

        var scale = (range.Max - range.Min) / (QMax - QMin);
        var bias = SMath.Round((range.Min * (QMin - QMax)) / (range.Max - range.Min)) + QMin;
        return new QuantParam((int)bias, (float)scale);
    }

    /// <summary>
    /// fixup range.
    /// </summary>
    /// <param name="range"></param>
    /// <param name="symmetric"></param>
    /// <returns></returns>
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
                r = 0.1f;
            else if (r < 0.01f)
                r = 0.01f;
            range.Max = range.Min + r;
        }

        return range;
    }

    public static ValueRange<T> GetRange<T>(Span<T> input) where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var data = input.ToArray();
        return new(data.Min(), data.Max());
    }

    public static List<float> GetWeightsRangesByChannel(Span<float> Weights, int Channels)
    {
        var tmpMin = float.MaxValue;
        var tmpMax = float.MinValue;
        var minMaxArr = new List<float>();
        for (int i = 0; i < Weights.Length; i++)
        {
            if (i % (Weights.Length / Channels) == 0)
            {
                tmpMin = float.MaxValue;
                tmpMax = float.MinValue;
            }

            if (Weights[i] < tmpMin)
                tmpMin = Weights[i];
            if (Weights[i] > tmpMax)
                tmpMax = Weights[i];
            if ((i + 1) % (Weights.Length / Channels) == 0)
            {
                minMaxArr.Add(tmpMin);
                minMaxArr.Add(tmpMax);
            }
        }

        return minMaxArr;
    }

    static void rounding_forward(float rounding_error_sum, ref torch.Tensor rounding_number_, ref torch.Tensor rounding_error_,
    torch.Tensor number_, torch.Tensor error_, ref torch.Tensor priority_, torch.Tensor order_, ref torch.Tensor priority_1)
    {
        int topk = (int)System.Math.Round(System.Math.Abs(rounding_error_sum));
        bool over_squant = (topk >= System.Math.Abs(rounding_error_sum));
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
    static void SQuant_func(torch.Tensor rounding_error_sum, ref torch.Tensor rounding_number, ref torch.Tensor rounding_error,
        torch.Tensor up_number, torch.Tensor up_error, ref torch.Tensor up_priority, torch.Tensor up_order,
        torch.Tensor down_number, torch.Tensor down_error, ref torch.Tensor down_priority, torch.Tensor down_order)
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
                    rounding_forward((float)rounding_error_sum[n][c], ref rounding_number_, ref rounding_error_,
                        up_number[n][c], up_error[n][c], ref priority_, up_order[n][c], ref priority_1);
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
                    rounding_forward((float)rounding_error_sum[n][c], ref rounding_number_, ref rounding_error_,
                        down_number[n][c], down_error[n][c], ref priority_, down_order[n][c], ref priority_1);
                    rounding_number[n][c] = rounding_number_;
                    rounding_error[n][c] = rounding_error_;
                    down_priority[n][c] = priority_;
                    up_priority[n][c] = priority_1;
                }
            }
        }
    }
    static torch.Tensor adaptive_round(torch.Tensor x, float t_min, float t_max)
    {
        bool squant_k = true;
        bool squant_c = true;

        var rounding_number = x.round(); //round取整值
        var rounding_error  = rounding_number - x; //误差
        var zeros = torch.zeros_like(rounding_error);

        var up_number = rounding_number.clone();
        var up_error  = rounding_error.clone();
        up_error = torch.where(x >= t_max, zeros, up_error); //边界上的值不能再调整，所以去除
        up_error = torch.where(up_error > 0, zeros, up_error); //误差为正的都设为0，即up对应“原值>量化值”的集合
        var up_priority = up_error.clone().abs();

        up_error = torch.where(up_error != 0, up_error + 1, up_error); //up集合中，Flip翻转后对应的误差
        up_number = torch.where(up_error != 0, up_number + 1, up_number); //up集合中，Flip翻转后对应的取整值

        var down_number = rounding_number.clone();
        var down_error  = rounding_error.clone();
        down_error = torch.where(x <= t_min, zeros, down_error); //边界上的值不能再调整，所以去除
        down_error = torch.where(down_error < 0, zeros, down_error); //误差为负的都设为0，即down对应“原值<量化值”的集合
        var down_priority = down_error.clone().abs();

        down_error = torch.where(down_error != 0, down_error - 1, down_error); //down集合中，Flip翻转后对应的误差
        down_number = torch.where(down_error != 0, down_number - 1, down_number); //down集合中，Flip翻转后对应的取整值

        var x_tmp = x.reshape(new long[]{x.size(0), x.size(1), -1});
        var conver_shape = x_tmp.shape; //HW维度合并
        if (conver_shape[2] == 1)
        {
            squant_k = false; //只有一个元素时， 不做K的逼近
        }
        if (squant_k)
        {
            var rounding_error_sum = rounding_error.reshape(conver_shape).sum(-1);
            var sort_ret = torch.sort(up_priority.reshape(conver_shape), -1, true);
            torch.Tensor up_order = sort_ret.Item2;
            sort_ret = torch.sort(down_priority.reshape(conver_shape), -1, true);
            torch.Tensor down_order = sort_ret.Item2;
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
            SQuant_func(rounding_error_sum, ref rounding_number, ref rounding_error, up_number, up_error, ref up_priority, up_order,
                down_number, down_error, ref down_priority, down_order);
            rounding_number = rounding_number.reshape(x.shape);
            rounding_error = rounding_error.reshape(x.shape);
            up_priority = up_priority.reshape(x.shape);
            down_priority = down_priority.reshape(x.shape);
        }

        if (squant_c)
        {
            conver_shape = new long[]{1, x.size(0), -1};
            var rounding_error_sum = rounding_error.reshape(conver_shape).sum(-1);
            var sort_ret = torch.sort(up_priority.reshape(conver_shape), -1, true);
            var up_order = sort_ret.Item2;
            sort_ret = torch.sort(down_priority.reshape(conver_shape), -1, true);
            var down_order = sort_ret.Item2;

            rounding_number = rounding_number.reshape(conver_shape);
            rounding_error = rounding_error.reshape(conver_shape);
            up_number = up_number.reshape(conver_shape);
            up_error = up_error.reshape(conver_shape);
            up_priority = up_priority.reshape(conver_shape);
            down_number = down_number.reshape(conver_shape);
            down_error = down_error.reshape(conver_shape);
            down_priority = down_priority.reshape(conver_shape);
            SQuant_func(rounding_error_sum, ref rounding_number, ref rounding_error, up_number, up_error, ref up_priority, up_order,
                down_number, down_error, ref down_priority, down_order);
        }
        rounding_number = rounding_number.reshape(x.shape);
        rounding_error = rounding_error.reshape(x.shape);
        up_priority = up_priority.reshape(x.shape);
        down_priority = down_priority.reshape(x.shape);

        return rounding_number;
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
        if (QuantMode.UnsignedMode == quantMode)
        {
            qmax = (1 << bits) - 1;
            qmin = 0;
        }
        else if (QuantMode.SignedAsymmetricMode == quantMode)
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
        if (4 == inputWeightsShape.Rank)
        {
            var out_channel = inputWeightsShape[0];
            var in_channel = inputWeightsShape[1];
            var filter_h = inputWeightsShape[2];
            var filter_w = inputWeightsShape[3];
            x = x.reshape(new long[]{out_channel.FixedValue, in_channel.FixedValue, filter_h.FixedValue, filter_w.FixedValue});
            if (isByChannel)
            {
                delta = torch.ones_like(x);
                zero_point = torch.zeros_like(x);
                for (var c = 0; c < out_channel.FixedValue; c++)
                {
                    var x_min = ((Tensor<float>)(((TensorConst)(inputWeightsRanges)).Value)).ToArray()[2 * c];
                    var x_max = ((Tensor<float>)(((TensorConst)(inputWeightsRanges)).Value)).ToArray()[2 * c + 1];
                    var delta_tmp = (x_max - x_min) / (qmax - qmin);
                    var zero_point_tmp = System.Math.Round(((x_max * qmin - x_min * qmax) / (x_max - x_min)));
                    delta[c] = torch.full_like(x[c], delta_tmp);
                    zero_point[c] = torch.full_like(x[c], zero_point_tmp);
                }
            }
            else
            {
                throw new Exception("By layer weights quant is not supported.");
            }
        }
        else
        {
            var out_channel = inputWeightsShape[0];
            var in_channel = inputWeightsShape[1];
            x = x.reshape(new long[]{out_channel.FixedValue, in_channel.FixedValue});
            if (isByChannel)
            {
                delta = torch.ones_like(x);
                zero_point = torch.zeros_like(x);
                for (var c = 0; c < out_channel.FixedValue; c++)
                {
                    var x_min = ((Tensor<float>)(((TensorConst)(inputWeightsRanges)).Value)).ToArray()[2 * c];
                    var x_max = ((Tensor<float>)(((TensorConst)(inputWeightsRanges)).Value)).ToArray()[2 * c + 1];
                    var delta_tmp = (x_max - x_min) / (qmax - qmin);
                    var zero_point_tmp = System.Math.Round(((x_max * qmin - x_min * qmax) / (x_max - x_min)));
                    delta[c] = torch.full_like(x[c], delta_tmp);
                    zero_point[c] = torch.full_like(x[c], zero_point_tmp);
                }
            }
            else
            {
                throw new Exception("By layer weights quant is not supported.");
            }
        }

        var quant_tensor = ((x / delta) + zero_point);
        var x_int = adaptive_round(quant_tensor, qmin, qmax); //SQuant量化
        var x_quant = torch.clamp(x_int, torch.tensor(qmin), torch.tensor(qmax));
        var x_dequant = (x_quant - zero_point) * delta;
        // var x_dequant = x_int; //x_int; //zero_point; //delta; //quant_tensor; //(x_quant - zero_point) * delta;

        Span<float> rst = new Span<float>(x_dequant.data<float>().ToArray());

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

    public static Span<float> AdaRoundWeights(Span<float> inputWeights, List<Tensor> layerInput, List<Tensor> layerOutputGT, QuantMode quantMode, int bits, bool isByChannel, Expr psum, Expr act, Expr paddings, Expr strides, Expr dilations, Expr groups, Expr fusedClamp, int startB, int endB, int iters, int deviceID, float warmup, float weightParam)
    {
        // todo: return AdaRoundWeights
        // System.Console.WriteLine(((TensorConst)(act)).Value.Cast<float>());
        return inputWeights;
    }
}