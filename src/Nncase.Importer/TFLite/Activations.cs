// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private static Expr Activate(Expr input, tflite.ActivationFunctionType activation)
        {
            return activation switch
            {
                tflite.ActivationFunctionType.NONE => input,
                _ => F.Math.Clamp(input, ToFloatValueRange(activation)),
            };
        }

        private static ValueRange<float> ToFloatValueRange(tflite.ActivationFunctionType activation)
        {
            return default;
        }

        private Expr VisitLogistic(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            return F.NN.Sigmoid(input);
        }

        private Expr VisitRelu(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            return F.NN.Relu(input);
        }

        private Expr VisitRelu6(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            return F.NN.Relu6(input);
        }

        private Expr VisitPRelu(in tflite.Operator op)
        {
            var (input, slope) = GetInputExprs(op, 0, 1);
            return F.NN.PRelu(input, slope);
        }

        private Expr VisitLeakyRelu(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            return F.NN.LeakyRelu(input, 0.01f);
        }
    }
}