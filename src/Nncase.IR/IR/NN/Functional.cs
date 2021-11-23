// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NN;

namespace Nncase.IR.F
{
    /// <summary>
    /// NN functional helper.
    /// </summary>
    public static class NN
    {

        public static Call Conv2D(Expr input, Expr weights, Expr bias, Expr stride, Expr padding, Expr dilation, PadMode padMode, Expr groups) => new Call(new Conv2D(padMode), input, weights, bias, stride, padding, dilation, groups);

        public static Call Conv2DTranspose(Expr input, Expr weights, Expr bias, Expr outShape, Expr padding, Expr stride, Expr dilation, PadMode padMode, Expr groups) => new Call(new Conv2DTranspose(padMode), input, weights, bias, outShape, padding, stride, dilation, groups);

        public static Call LeakyRelu(Expr input) => new Call(new LeakyRelu(), input);

        public static Call L2Normalization(Expr input) => new Call(new L2Normalization(), input);

        public static Call Relu(Expr input) => new Call(new Relu(), input);

        public static Call Relu6(Expr input) => new Call(new Relu6(), input);

        public static Call PRelu(Expr input) => new Call(new PRelu(), input);

        public static Call Sigmoid(Expr expr) => new Call(new Sigmoid(), expr);

        public static Call SoftMax(Expr expr) => new Call(new SoftMax(), expr);

        public static Call LogSoftMax(Expr expr) => new Call(new LogSoftMax(), expr);
    }
}
