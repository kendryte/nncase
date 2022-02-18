// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Pattern;
using Nncase.Pattern.Math;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.Pattern.F.Tensors;
using Nncase.Pattern.Tensors;
using System.Numerics.Tensors;

namespace Nncase.Transform.Rule
{
    public class SpaceToBatchToPad : IRewriteRule
    {
        private SpaceToBatchWrapper s2b;

        public SpaceToBatchToPad()
        {
            Pattern = s2b = SpaceToBatch(IsWildcard(), IsConst(), IsConst());
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            s2b.Bind(result);
            var block_shape = s2b.BlockShape<TensorConst>().Value.Cast<int>();
            if (block_shape[0] == 1 && block_shape[1] == 1)
            {
                var pads = s2b.Paddings<TensorConst>().Value.Cast<int>();
                var newpads = Tensor.FromScalar(0, new[] { 3, 2 });
                newpads[2, 0] = pads[0];
                newpads[2, 1] = pads[1];
                return Pad(s2b.Input(), newpads, PadMode.Constant, 0);
            }

            return null;
        }
    }
}