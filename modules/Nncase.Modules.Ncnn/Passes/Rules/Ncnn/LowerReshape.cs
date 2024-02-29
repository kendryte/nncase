// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerReshape : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsReshape(
        IsWildcard("input"),
        IsTensorConst("shape"));

    private Expr? GetReplace(Expr input, Expr shape)
    {
        if (input.CheckedShape.Count < 5)
        {
            var inResO = new Var(input.CheckedType);
            var outputShape = shape.Evaluate().AsTensor().ToArray<int>();

            return new Call(new Fusion("ncnn", NcnnReshape(inResO, outputShape), new[] { inResO }), input);
        }

        return null;

        // // TODO: split input
        // // if shape.Length == input.Length means that batchSize need reshape. Ncnn can't support.
        // if (input.CheckedShape.ToList()[0] != 1 || input.CheckedShape.Count == shape.CheckedShape.Count)
        // {
        //     return null;
        // }
        //
        // var inRes = Squeeze(input, new[] { 0 });
        // var inResO = new Var(inRes.CheckedType);
        // var outputShape = shape.Evaluate().AsTensor().ToArray<int>();
        // var reshape = new Call(new Fusion("ncnn", NcnnReshape(inResO, outputShape), new[] { inResO }), input);
        //
        // return reshape;
    }
}
