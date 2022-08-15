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
using static  Nncase.PatternMatch.F.K210;
using Nncase.PatternMatch;
using Nncase.Utilities;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Math = Nncase.IR.F.Math;
namespace Nncase.Transform.Rules;

/// <summary>
/// Lower <see cref="IR.K210.KPUUpload"/> to <see cref="IR.K210.FoldKPUUpload"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldKPUDownload : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsKPUDownload(
        null,
        "upload_call",
        IsWildcard("input"));

    private Expr? GetReplace(Call upload_call, Expr input)
    {
        return input;
    }
}