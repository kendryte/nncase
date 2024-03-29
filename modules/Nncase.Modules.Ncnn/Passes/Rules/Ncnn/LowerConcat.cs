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
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerConcat : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsConcat(
        "concat",
        _ => true,
        IsTuple("tuple", IsVArgsRepeat("tupleInputs", () => IsWildcard() with { TypePattern = HasFixedShape() })));

    // squeeze concat to 3D, get outputShape，set new axis as 1
    private static (List<List<int>> NewShape, List<int> OldOutputShape) GetFixedShapeAndOldOutputShape(IReadOnlyList<Expr> tupleInputs, int oldAxis)
    {
        var newShapes = new List<List<int>>();
        var outputShape = tupleInputs[0].CheckedShape.ToValueList();
        int positive_axis = oldAxis < 0 ? tupleInputs[0].CheckedShape.Count + oldAxis : oldAxis;
        outputShape[positive_axis] = 0;
        foreach (var input in tupleInputs)
        {
            var oldInputShape = input.CheckedShape.ToValueList();
            outputShape[positive_axis] += oldInputShape[positive_axis];
            var newShape = new List<int> { 1, oldInputShape[positive_axis], 1 };
            for (int i = 0; i < positive_axis; i++)
            {
                newShape[0] *= oldInputShape[i];
            }

            for (int i = positive_axis + 1; i < oldInputShape.Count; i++)
            {
                newShape[2] *= oldInputShape[i];
            }

            newShapes.Add(newShape);
        }

        return (newShapes, outputShape);
    }

    private Expr? GetReplace(IReadOnlyList<Expr> tupleInputs, IR.Tensors.Concat concat)
    {
        var (newInputShapes, oldOutputShape) = GetFixedShapeAndOldOutputShape(tupleInputs, concat.Axis);

        var varOfNewInputs = new List<Var>();
        var callOfNewInputs = new List<Call>();
        for (int i = 0; i < tupleInputs.Count; i++)
        {
            var inRes = Reshape(tupleInputs[i], newInputShapes[i].ToArray());
            var inResO = new Var(inRes.CheckedType);
            varOfNewInputs.Add(inResO);
            callOfNewInputs.Add(inRes);
        }

        var ncnnConcatCall = new Call(new Fusion("ncnn", NcnnConcat(varOfNewInputs.ToArray(), 1), varOfNewInputs.ToArray()), callOfNewInputs.ToArray());

        var outRes = Reshape(ncnnConcatCall, oldOutputShape.ToArray());

        return outRes;
    }
}
