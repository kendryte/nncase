// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using Nncase.Pattern.Tensors;
using Nncase.Pattern.NN;
using Nncase.Pattern.Math;
using Nncase.Pattern;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public sealed class AddToConv2D : PatternRule
    {
        private BinaryWrapper ad;

        public AddToConv2D()
        {
            Pattern = ad = Add(IsWildCard(), IsWildCard());
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            ad.Bind(result);
            var a_sp = ad.Lhs().CheckedShape;
            var b_sp = ad.Rhs().CheckedShape;
            if (a_sp.Rank == 4 && a_sp == b_sp)
            {
                var channels = a_sp[1].FixedValue;
                var weights = new Tensor<float>(channels * 2 * channels);
                for (int i = 0; i < channels; i++)
                {
                    weights[(2 * channels * i) + i] = 1.0f;
                    weights[(2 * channels * i) + i + channels] = 1.0f;
                }

                var c = Concat(new IR.Tuple(ad.Lhs(), ad.Rhs()), 1);
                var con_weights = Const.FromTensor(weights.Reshape(new[] { channels, 2 * channels, 1, 1 }));

                return Conv2D(
                    c,
                    con_weights,
                    Tensor.FromScalar(0.0f, channels),
                    Tensor.FromSpan(new[] { 0, 0, 0, 0 }, new[] { 2, 2 }),
                    new[] { 1, 1 },
                    new[] { 1, 1 },
                    PadMode.Constant,
                    1);
            }

            return null;
        }
    }
}
