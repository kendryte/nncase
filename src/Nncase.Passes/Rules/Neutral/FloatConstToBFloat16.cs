// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold call of constants.
/// </summary>
[RuleGenerator]
public partial class FloatConstToBFloat16 : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsTensorConst("tc", HasDataType(DataTypes.Float32) & HasRank(2));

    private Expr GetReplace(TensorConst tc)
    {
        // note for egraphs.
        var newTc = new TensorConst(tc.Value.CastTo(DataTypes.BFloat16));
        var cast = Nncase.IR.F.Tensors.Cast(newTc, DataTypes.Float32);
        return cast.InheritMetaData(tc);
    }
}
