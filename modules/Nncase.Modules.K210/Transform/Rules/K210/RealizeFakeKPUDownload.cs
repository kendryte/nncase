// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K210;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Utilities;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.K210;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Math = Nncase.IR.F.Math;

namespace Nncase.Passes.Rules.K210;

/// <summary>
/// Lower <see cref="IR.K210.FakeKPUDownload"/> to <see cref="IR.K210.KPUDownload"/>.
/// </summary>
[RuleGenerator]
public sealed partial class RealizeFakeKPUDownload : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFakeKPUDownload(
            null,
            "download_call",
            op => true,
            IsRangeOfMarker(
                IsWildcard("input"),
                IsConst("input_range")));

    private Expr? GetReplace(Call download_call, Expr input, Expr input_range)
    {
        // return new Function(IR.F.K210.KPUDownload(input));
        return new Call(new IR.K210.KPUDownload(), input);
    }
}
