// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using tflite;
using static Nncase.IR.F.NN;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite;

public partial class TFLiteImporter
{
    private static Expr Activate(Expr input, tflite.ActivationFunctionType activation)
    {
        _ = ToFloatValueRange(activation);
        return activation switch
        {
            tflite.ActivationFunctionType.NONE => input,
            tflite.ActivationFunctionType.RELU => Relu(input),

            // ActivationFunctionType.RELU_N1_TO_1 => expr,
            ActivationFunctionType.RELU6 => Relu6(input),
            ActivationFunctionType.TANH => F.Math.Tanh(input),

            // ActivationFunctionType.SIGN_BIT => expr,
            _ => throw new NotImplementedException(activation.ToString()),
        };
    }

    private static ValueRange<float> ToFloatValueRange(tflite.ActivationFunctionType activation)
    {
        return default;
    }

    private Expr VisitLogistic(in tflite.Operator op)
    {
        var input = GetInputExprs(op, 0);
        return SetOutputsNames(
            F.NN.Sigmoid(input),
            1,
            op);
    }

    private Expr VisitRelu(in tflite.Operator op)
    {
        var input = GetInputExprs(op, 0);
        return SetOutputsNames(
            F.NN.Relu(input),
            1,
            op);
    }

    private Expr VisitRelu6(in tflite.Operator op)
    {
        var input = GetInputExprs(op, 0);
        return SetOutputsNames(
            F.NN.Relu6(input),
            1,
            op);
    }

    private Expr VisitPRelu(in tflite.Operator op)
    {
        var (input, slope) = GetInputExprs(op, 0, 1);
        return SetOutputsNames(
            F.NN.PRelu(input, slope),
            1,
            op);
    }

    private Expr VisitLeakyRelu(in tflite.Operator op)
    {
        var input = GetInputExprs(op, 0);
        return SetOutputsNames(
            F.NN.LeakyRelu(input, op.BuiltinOptionsAsLeakyReluOptions().Alpha),
            1,
            op);
    }
}
