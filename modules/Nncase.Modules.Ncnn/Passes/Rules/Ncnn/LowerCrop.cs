// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.ArgsStruct;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerCrop : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSlice(
      IsWildcard("input"),
      IsTensorConst("starts"),
      IsTensorConst("ends"),
      IsTensorConst("axes"),
      IsTensorConst("steps"));

    private Expr? GetReplace(Expr input, int[] starts, int[] ends, int[] axes, int[] steps)
    {
        if (steps.All(x => x != 1))
        {
            Console.WriteLine("ncnn not support slice with step");
            return null;
        }

        if (input.CheckedShape.Count > 4 || input.CheckedShape[0].FixedValue != 1)
        {
            Console.WriteLine("ncnn not support more than 4D or batchSize > 1");
            return null;
        }

        var tStart = starts.ToList();
        var tEnds = ends.ToList();
        var tAxes = axes.ToList();

        for (int i = 0; i < axes.Length; i++)
        {
            if (axes[i] == 0)
            {
                tStart.RemoveAt(i);
                tEnds.RemoveAt(i);
                tAxes.RemoveAt(i);
            }

            break;
        }

        for (int i = 0; i < tAxes.Count; i++)
        {
            tAxes[i] -= 1;
        }

        var args = new CropArgs(tStart.ToArray(), tEnds.ToArray(), tAxes.ToArray());

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        var crop = new Call(new Fusion("ncnn", NcnnCrop(inResO, args), new[] { inResO }), inRes);
        return Unsqueeze(crop, new[] { 0 });
    }
}
