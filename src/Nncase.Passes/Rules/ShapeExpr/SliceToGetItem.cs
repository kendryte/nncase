using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Passes.Rules.Neutral.FoldSqueezeCommon;

namespace Nncase.Passes.Rules.Neutral;

// Slice(shape, 1, 2, 1, 1) -> shape[1]
[RuleGenerator]
public partial class SliceToGetItem : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsSqueeze(
        IsSlice(
            IsWildcard("input") with { TypePattern = HasRank(1) },
            IsTensorConst("begins"),
            IsTensorConst("ends"),
            IsTensorConst("axes"),
            IsTensorConst("strides", strides => strides.Value.ToArray<int>()[0] == 1)),
        IsTensorConst("dims"));

    Expr? GetReplace(Expr input, int[] begins, int[] ends)
    {
        if ((ends[0] - begins[0]) == 1)
        {
            return input[begins[0]];
        }

        return null;
    }
}
