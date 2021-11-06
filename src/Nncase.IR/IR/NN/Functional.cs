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
        /// <summary>
        /// Call sigmoid.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Sigmoid(Expr expr) => new Call(new Sigmoid(), expr);

        public static Call Conv2D(Expr input, Expr weights, Expr bias, Expr padding, Expr stride, Expr dilation, PadMode padMode) => new Call(new Conv2D(padMode), input, padding, stride, dilation);
        
        public static Call Conv2DTranspose(Expr input, Expr weights, Expr bias, Expr padding, Expr stride, Expr dilation, PadMode padMode) => new Call(new Conv2DTranspose(padMode), input, padding, stride, dilation);
    }
}
