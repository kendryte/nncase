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
using Nncase.IR.NN;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ShapeUtility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerLayerNorm : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsLayerNorm(
        "op",
        _ => true,
        IsWildcard("input"),
        IsTensorConst("gamma"),
        IsTensorConst("beta"));

    private Expr? GetReplace(LayerNorm op, Expr input, float[] gamma, float[] beta)
    {
        // TODO: split input
        if (input.CheckedShape.ToList()[0] != 1)
        {
            return null;
        }

        // var newShape = FitNcnnShape(input.CheckedShape.ToValueList(), op.Axis); // newAxis = 1;
        // Console.WriteLine($"op.Axis: {op.Axis}");
        // Console.WriteLine($"oldShape: {input.CheckedShape.ToValueList().Select(x => x.ToString()).Aggregate((x, y) => x + ", " + y)}");
        // Console.WriteLine($"newShape: {newShape.Select(x => x.ToString()).Aggregate((x, y) => x + "," + y)}");
        // var inRes = Reshape(input, newShape.ToArray());
        // var inResO = new Var(inRes.CheckedType);

        // int affine = gamma.All(a => a != 1) && beta.All(a => a != 0) ? 1 : 0;
        // int affineSize = newShape[2];

        // var layerNorm = new Call(new Fusion("ncnn", NcnnLayerNorm(inResO, affineSize, op.Epsilon, affine, gamma, beta), new[] { inResO }), inRes);

        // return Reshape(layerNorm, input.CheckedShape);

        // var newShape = FitNcnnShape(input.CheckedShape.ToValueList(), op.Axis); // newAxis = 1;
        // Console.WriteLine($"op.Axis: {op.Axis}");
        // Console.WriteLine($"oldShape: {input.CheckedShape.ToValueList().Select(x => x.ToString()).Aggregate((x, y) => x + ", " + y)}");
        // Console.WriteLine($"newShape: {newShape.Select(x => x.ToString()).Aggregate((x, y) => x + "," + y)}");
        var inRes = Squeeze(input, new[]{0});
        var inResO = new Var(inRes.CheckedType);

        int affine = gamma.All(a => a != 1) && beta.All(a => a != 0) ? 1 : 0;
        int affineSize = 1;
        for(int i = 1; i < inRes.CheckedShape.Rank; i++)
        {
            affineSize *= inRes.CheckedShape.ToValueArray()[i];
        }

        var layerNorm = new Call(new Fusion("ncnn", NcnnLayerNorm(inResO, affineSize, op.Epsilon, affine, gamma, beta), new[] { inResO }), inRes);

        return Unsqueeze(layerNorm, new[] { 0 });
    }
}
