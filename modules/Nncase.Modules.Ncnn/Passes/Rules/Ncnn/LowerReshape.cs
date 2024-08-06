// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerReshape : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsReshape(
        IsWildcard("input") with { TypePattern = HasFixedShape() & !IsScalar() },
        IsTensorConst("shape"));

    private Expr? GetReplace(Expr input, Expr shape)
    {
        if (input.CheckedShape.Rank < 5)
        {
            Call r;
            var newInput = input;
            var newInputVar = new Var(input.CheckedType);

            var outputShape = shape.Evaluate().AsTensor().ToArray<int>().ToList();

            int needSqueeze = 0;
            if (newInput.CheckedShape[0].FixedValue != 1 && newInput.CheckedShape.Rank == 4)
            {
                return null;
            }
            else if (newInput.CheckedShape[0].FixedValue == 1 && newInput.CheckedShape.Rank == 4)
            {
                newInput = Squeeze(newInput, new[] { 0 });
                newInputVar = new Var(newInput.CheckedType);
                needSqueeze += 1;
            }

            if (outputShape.Count > 4 || (outputShape.Count == 4 && outputShape[0] != 1))
            {
                return null;
            }
            else if (outputShape.Count == 4 && outputShape[0] == 1 && needSqueeze == 1)
            {
                outputShape.RemoveAt(0); // Avoid reshape input to 4D with multi batchsize.
                needSqueeze += 2;
            } // left 3D shape.

            if (outputShape.Count == 0)
            {
                return null;
            }

            r = new Call(new Fusion("ncnn", NcnnReshape(newInputVar, outputShape.ToArray()), new[] { newInputVar }), newInput);

            if (needSqueeze == 3)
            {
                return Unsqueeze(r, new[] { 0 });
            }

            return r;
        }

        return null;
    }
}
