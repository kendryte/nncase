// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

/// <summary>
/// Convert QKV computation to MHA style.
/// </summary>
[RuleGenerator]
public sealed partial class MHACombine : IRewriteRule
{
    public IPattern Pattern { get; } =
        IsTranspose(
        "tp",
        "tpCall",
        _ => true,
        IsReshape(
            "rshape",
            "rshapeCall",
            _ => true,
            IsMatMul(
                "mm",
                "mmCall",
                _ => true,
                IsWildcard("x"),
                IsTensorConst("w")),
            IsTensorConst("newShape")),
        IsTensorConst("perm")) with
        { TypePattern = HasFixedShape() };

    private Expr? GetReplace(Expr x, Call mmCall, Call rshapeCall, Call tpCall, TensorConst w, TensorConst newShape, TensorConst perm)
    {
        var mmOutShape = mmCall.CheckedShape.ToValueArray();
        var wReshape = newShape.Value.ToArray<int>().TakeLast(2).ToArray();

        // TODO: add more patterns, only llama65b for now
        if (perm.CheckedShape.Size == 4
            && perm.Value.ToArray<int>().SequenceEqual(new[] { 0, 2, 1, 3 })
             && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
             && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = Tensors.Transpose(Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = Tensors.MatMul(Tensors.Unsqueeze(x, new[] { 1 }), newW);
            return newMm;
        }

        return null;
    }
}
