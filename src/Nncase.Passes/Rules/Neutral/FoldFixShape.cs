// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// FoldFixShape <see cref="IR.Tensors.Cast"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldFixShape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCall(new OpPattern<FixShape>(x => true, string.Empty), IsWildcard("input"), IsWildcard());

    private Expr? GetReplace(Expr input)
    {
        return input;
    }
}
