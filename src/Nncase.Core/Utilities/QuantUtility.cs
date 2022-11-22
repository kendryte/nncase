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

    public static Span<float> SquantWeights(Span<float> inputWeights, Expr inputWeightsRanges, Nncase.IR.Shape inputWeightsShape, QuantMode quantMode, int bits, bool isByChannel)
    {
        // todo: return SquantWeights
        // inputWeightsRanges is pre calculated by range optimization, so when compute inputWeights quant parameters, range should be gotten from inputWeightsRanges, but not
        // be gotten from inputWeights here simply. And for quantMode, there are 3 modes, UnsignedMode is easy to understand, and for SignedAsymmetricMode/SignedSymmetricMode,
        // it effects Qmax/Qmin for quant function, for example, k510 int8 needs SignedAsymmetricMode, and k230 int8/int16 needs SignedSymmetricMode, please refer GetQuantParam() in this file.

        // System.Console.WriteLine(((Tensor<float>)(((TensorConst)(inputWeightsRanges)).Value)).ToArray()[0]);
        // System.Console.WriteLine(inputWeights.ToArray()[0]);
        return inputWeights;
    }

    public static Span<float> AdaRoundWeights(Span<float> inputWeights, List<Tensor> layerInput, List<Tensor> layerOutputGT, QuantMode quantMode, int bits, bool isByChannel, Expr psum, Expr act, Expr paddings, Expr strides, Expr dilations, Expr groups, Expr fusedClamp, int startB, int endB, int iters, int deviceID, float warmup, float weightParam)
    {
        // todo: return AdaRoundWeights
        // System.Console.WriteLine(((TensorConst)(act)).Value.Cast<float>());
        return inputWeights;
    }
}