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
/// %9 = MatMul(%2, const(f32[768,768]))
/// %10 = Add(BinaryOp.Add, const(f32[768]), %9)
/// %11 = Reshape(%10, const(i32[4] : {1,77,12,64}))
/// %12 = Transpose(%11, const(i64[4] : {0L,2L,1L,3L}))
/// %13 = Reshape(%12, const(i32[3] : {12,77,64})).
/// </summary>
[RuleGenerator]
public sealed partial class CombineMHA : IRewriteRule
{
    public CombineMHA()
    {
        Pattern v0 = IsMatMul("mm", "mmCall", IsWildcard("x"), IsTensorConst("w"));

        var bias = IsAlt(
            IsBinary("add", "addCall", op => op.BinaryOp == BinaryOp.Add, IsTensorConst("bias"), v0),
            IsBinary("add", "addCall", op => op.BinaryOp == BinaryOp.Add, v0, IsTensorConst("bias")),
            v0);
        var scale = IsAlt(
            IsBinary("mul", "mulCall", op => op.BinaryOp == BinaryOp.Mul, bias, IsTensorConst("scale")),
            IsBinary("mul", "mulCall", op => op.BinaryOp == BinaryOp.Mul, IsTensorConst("scale"), bias),
            bias);

        var v1 = IsReshape("rshape", "rshapeCall", scale, IsTensorConst("newShape"));
        var v2 = IsTranspose("tp", "tpCall", v1, IsTensorConst("perm")) with { TypePattern = HasFixedShape() };
        Pattern = v2;
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x, Call mmCall, TensorConst w, TensorConst newShape, int[] perm, IMatchResult matchResult)
    {
        var mmOutShape = mmCall.CheckedShape.ToValueArray();
        var wReshape = newShape.Value.ToArray<int>().TakeLast(2).ToArray();

        // TODO: add more patterns, only llama65b for now
        if (perm.Length == 4 && perm.SequenceEqual(new[] { 0, 2, 1, 3 })
             && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
             && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = Tensors.Transpose(Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = Tensors.MatMul(Tensors.Unsqueeze(x, new[] { 1 }), newW);
            if (matchResult["bias"] is TensorConst bias)
            {
                return null;
            }

            if (matchResult["scale"] is TensorConst scale)
            {
                return null;
            }

            return newMm;
        }
        else if (perm.Length == 3 && perm.SequenceEqual(new[] { 1, 0, 2 })
            && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
            && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = Tensors.Transpose(Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = Tensors.MatMul(x, newW);
            if (matchResult["bias"] is TensorConst bias)
            {
                newMm = IR.F.Math.Add(newMm, bias.Value.Shape.IsScalar ? bias : Tensors.Reshape(bias, new[] { -1, 1, wReshape[1] }));
            }

            if (matchResult["scale"] is TensorConst scale)
            {
                newMm = IR.F.Math.Mul(newMm, scale.Value.Shape.IsScalar ? scale : Tensors.Reshape(scale, new[] { -1, 1, wReshape[1] }));
            }

            return newMm;
        }
        else if (perm.Length == 3 && perm.SequenceEqual(new[] { 1, 2, 0 })
            && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
            && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = Tensors.Transpose(Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = Tensors.MatMul(x, newW);
            if (matchResult["bias"] is TensorConst bias)
            {
                newMm = IR.F.Math.Add(newMm, bias.Value.Shape.IsScalar ? bias : Tensors.Reshape(bias, new[] { -1, 1, wReshape[1] }));
            }

            if (matchResult["scale"] is TensorConst scale)
            {
                newMm = IR.F.Math.Mul(newMm, scale.Value.Shape.IsScalar ? scale : Tensors.Reshape(scale, new[] { -1, 1, wReshape[1] }));
            }

            return Tensors.Transpose(newMm, new[] { 0, 2, 1 });
        }

        return null;
    }
}
