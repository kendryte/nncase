// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Unsqueeze = Nncase.IR.Tensors.Unsqueeze;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerSlice : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSplit(
      IsWildcard("input") with { TypePattern = HasFixedShape() },
      IsTensorConst("axis") with { TypePattern = IsScalar() },
      IsTensorConst("slices"));

    private Expr? GetReplace(Expr input, int[] slices, int axis)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        if (input.CheckedShape.Count <= 5 && axis != 0)
        {
            var inRes = Squeeze(input, new[] { 0 });
            var inResO = new Var(inRes.CheckedType);
            var newAxis = axis < 0 ? axis + input.CheckedShape.Count - 1 : axis - 1;
            var slice = new Call(new Fusion("ncnn", NcnnSlice(inResO, slices, newAxis), new[] { inResO }), inRes);
            var result = new List<Expr>();
            for (int i = 0; i < slices.Length; i++)
            {
                result.Add(Unsqueeze(slice[i], new[] { 0 }));
            }

            return new IR.Tuple(result.ToArray());
        }

        return null;
    }
}
