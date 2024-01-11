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
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ShapeUtility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerCumsum : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsCumSum(
      IsWildcard("input"),
      IsTensorConst("axis"),
      IsTensorConst("exclusive"),
      IsTensorConst("reverse"));

    private Expr? GetReplace(Expr input, Tensor<int> axis, Tensor<bool> exclusive, Tensor<bool> reverse)
    {
        if (exclusive[0] == false && reverse[0] == false)
        {
            var inResShape = FitNcnnShape(input.CheckedShape.ToValueList(), axis[0]);
            var inRes = Reshape(input, inResShape.ToArray());
            var inRes0 = new Var(inRes.CheckedType);

            var newInput = new Var(input.CheckedType);
            var cumsum_ = new Call(new Fusion("ncnn", NcnnCumsum(newInput, 1), new[] { newInput }), inRes);

            var outRes = Reshape(cumsum_, input.CheckedShape.ToValueList().ToArray());

            return outRes;
        }

        return null;
    }
}
