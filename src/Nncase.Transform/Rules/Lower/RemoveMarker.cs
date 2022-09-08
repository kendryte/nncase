// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics.Tensors;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Transform;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Lower;

[RuleGenerator]
public sealed partial class RemoveMarker : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsMarker(null, s => true, IsWildcard("target"), IsWildcard("attribute"));

    private Expr? GetReplace(Expr target, Expr attribute)
    {
        return target;
    }
}
