// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using Nncase.Pattern.Tensors;
using Nncase.Pattern.NN;
using Nncase.Pattern;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldPadConv2d : PatternRule
    {
        Conv2DWrapper conv2d;
        PadWrapper pad;
        public FoldPadConv2d()
        {
            pad = IsPad(IsWildCard(), IsConst(), PadMode.Constant, IsConst());
            Pattern = conv2d = IsConv2D(pad, PadMode.Constant);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            pad.Bind(result);
            conv2d.Bind(result);
            var pads = pad.Pads<TensorConst>().Value.Cast<int>();
            var padv = pad.Value<TensorConst>().Value.ToScalar<float>();
            if (pads.Dimensions[0] == 4
            && pads[2, 0] >= 0 && pads[2, 1] >= 0
            && pads[3, 0] >= 0 && pads[3, 1] >= 0
            && ((pads[2, 0] + pads[2, 1]) > 0 || (pads[3, 0] + pads[3, 1]) > 0)
            && padv == .0f)
            {
                var newpads = new Tensor<int>(new[] { 2, 2 });
                for (int i = 2; i < 4; i++)
                {
                    if (pads[i, 0] > 0)
                    {
                        newpads[i - 2, 0] += pads[i, 0];
                        pads[i, 0] = 0;
                    }

                    if (pads[i, 1] > 0)
                    {
                        newpads[i - 2, 1] += pads[i, 1];
                        pads[i, 1] = 0;
                    }
                }

                return Conv2D(Pad(pad.Input(), Const.FromTensor(pads), pad.PadMode, pad.Value()), conv2d.Weights(), conv2d.Bias(), Const.FromTensor(newpads), conv2d.Stride(), conv2d.Dilation(), conv2d.PadMode, conv2d.Groups());
            }

            return null;
        }
    }
}