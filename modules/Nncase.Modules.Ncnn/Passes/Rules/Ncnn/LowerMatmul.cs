// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerMatmul : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsMatMul(
        IsWildcard("inputA") with { TypePattern = HasFixedShape() },
        IsWildcard("inputB") with { TypePattern = HasFixedShape() });

    private Expr? GetReplace(Expr inputA, Expr inputB)
    {
        if (inputA is Const)
        {
            var constA = ((TensorConst)inputA).Value;
            var constShape = inputA.CheckedShape.ToValueArray();

            // var newB = Reshape(inputB, FixShape(inputB.CheckedShape.ToValueArray(), r));
            var newInputB = new Var(inputB.CheckedType);
            return new Call(new Fusion("ncnn", NcnnMatMul(new Expr[] { newInputB }, 1, constA.ToArray<float>(), constShape), new[] { newInputB }), inputB);
        }

        if (inputB is Const)
        {
            // var newA = Reshape(inputA, FixShape(inputA.CheckedShape.ToValueArray(), r));
            var newInputA = new Var(inputA.CheckedType);
            var constB = ((TensorConst)inputB).Value;
            var constShape = inputB.CheckedShape.ToValueArray();
            return new Call(new Fusion("ncnn", NcnnMatMul(new Expr[] { newInputA }, 2, constB.ToArray<float>(), constShape), new[] { newInputA }), inputA);
        }

        {
            var newInputA = new Var(inputA.CheckedType);
            var newInputB = new Var(inputB.CheckedType);
            return new Call(
                new Fusion("ncnn", NcnnMatMul(new Expr[] { newInputA, newInputB }, 0, null, null), new[] { newInputA, newInputB }), inputA, inputB);
        }
    }
}
