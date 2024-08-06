// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerSoftmax : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSoftmax(
      IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() },
      IsTensorConst("axis"));

    // squeeze softmax to 3D，set axis to 1
    private (List<int> NewShape, int NewAxis) GetFixedShapeAndAxis(List<int> oldShape, int oldAxis)
    {
        int positive_axis = oldAxis < 0 ? oldShape.Count + oldAxis : oldAxis;
        var newShape = new List<int> { 1, oldShape[positive_axis], 1 };
        for (int i = 0; i < positive_axis; i++)
        {
            newShape[0] *= oldShape[i];
        }

        for (int i = positive_axis + 1; i < oldShape.Count; i++)
        {
            newShape[2] *= oldShape[i];
        }

        return (newShape, 1);
    }

    private Expr? GetReplace(Expr input, int axis)
    {
        var newInput = new Var(input.CheckedType);
        if (input.CheckedShape.Rank > 3)
        {
            var (newShape, newAxis) = GetFixedShapeAndAxis(input.CheckedShape.ToValueList(), axis);

            var inRes = Reshape(input, newShape.ToArray());
            var inResO = new Var(inRes.CheckedType);

            var ncnnSoftmaxCall = new Call(new Fusion("ncnn", NcnnSoftmax(inResO, newAxis), new[] { inResO }), inRes);

            var outRes = Reshape(ncnnSoftmaxCall, input.CheckedShape.ToValueList().ToArray());
            return outRes;
        }
        else
        {
            return new Call(new Fusion("ncnn", NcnnSoftmax(newInput, axis), new[] { newInput }), input);
        }
    }
}
