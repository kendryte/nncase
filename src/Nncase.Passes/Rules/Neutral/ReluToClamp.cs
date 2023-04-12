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
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// convert <see cref="IR.NN.Relu"/> to <see cref="IR.Math.Clamp"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ReluToClamp : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsRelu(IsWildcard("input") with { TypePattern = HasDataType(DataTypes.Float32) });

    private Expr? GetReplace(Expr input)
    {
        return Clamp(input, 0.0f, float.PositiveInfinity);
    }
}

/// <summary>
/// convert <see cref="IR.NN.Relu6"/> to <see cref="IR.Math.Clamp"/>.
/// </summary>
[RuleGenerator]
public sealed partial class Relu6ToClamp : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } = IsRelu6(IsWildcard("input") with { TypePattern = HasDataType(DataTypes.Float32) });

    private Expr? GetReplace(Expr input)
    {
        return Clamp(input, 0.0f, 6.0f);
    }
}
