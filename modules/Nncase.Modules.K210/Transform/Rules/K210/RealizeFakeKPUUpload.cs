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
using Tensorflow.Contexts;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.K210;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Math = Nncase.IR.F.Math;

namespace Nncase.Passes.Rules.K210;

/// <summary>
/// Lower <see cref="IR.K210.FakeKPUUpload"/> to <see cref="IR.K210.KPUUpload"/>.
/// </summary>
[RuleGenerator]
public sealed partial class RealizeFakeKPUUpload : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFakeKPUUpload(
            null,
            "upload_call",
            IsWildcard("input"));

    private Expr? GetReplace(Call upload_call, Expr input)
    {
        return IR.F.K210.KPUUpload(input);

        // var inputVar = new Var();
        //  var func = new Function(
        //     new Call(new IR.K210.KPUUpload(), inputVar), new[]{inputVar});
        // return new Call(null);
    }
}
