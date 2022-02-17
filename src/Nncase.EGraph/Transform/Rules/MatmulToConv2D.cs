// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using Nncase.Pattern.Tensors;
using Nncase.Pattern.NN;
using Nncase.Pattern;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class MatMulToConv2D : IRewriteRule
    {
        MatMulWrapper matmul;
        public MatMulToConv2D()
        {
            Pattern = matmul = MatMul(IsWildCard(), IsWildCard());
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            matmul.Bind(result);
            var input_shape = matmul.Input().CheckedShape;
            var other_shape = matmul.Other().CheckedShape;
            var if_shape = new Shape(new[] { input_shape[0].FixedValue, input_shape[1].FixedValue, 1, 1 });
            var w_shape = new Shape(new[] { other_shape[1].FixedValue, other_shape[0].FixedValue, 1, 1 });

            var if_reshape = Reshape(matmul.Input(), if_shape);
            var w_tp = Transpose(matmul.Other(), Tensor.FromSpan<int>(new[] { 1, 0 }));
            var w_reshape = Reshape(w_tp, w_shape);

            return Conv2D(
              if_reshape,
              w_reshape,
              Tensor.FromScalar(0.0f, input_shape[1].FixedValue),
              Tensor.FromScalar(0, new[] { 2, 2 }),
              new int[] { 1, 1 },
              new int[] { 1, 1 },
              PadMode.Constant,
              1);
        }
    }
}