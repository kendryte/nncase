// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Base function.
/// </summary>
public class BaseFunction
{
    public static Expr? BroadcastOutputNames(Call call, Expr expr)
    {
        if (call.Metadata.OutputNames == null)
        {
            if (expr.Metadata.OutputNames != null && expr.Metadata.OutputNames!.Count == 1)
            {
                call.Metadata.OutputNames = expr.Metadata.OutputNames;
                return null;
            }
        }
        else
        {
            if (expr.Metadata.OutputNames == null)
            {
                expr.Metadata.OutputNames = call.Metadata.OutputNames;
                return null;
            }
        }

        return null;
    }
}

/// <summary>
/// Broadcast nop pad outputs names.
/// </summary>
[RuleGenerator]
public sealed partial class BroadcastNopPadOutputNames : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsPad("pad", "padCall", padMode => true, IsWildcard("input"), IsTensorConst("pads"), IsWildcard("padValue"));

    private Expr? GetReplace(Call padCall, Expr input, TensorConst pads, Expr padValue)
    {
        if (pads.Value.Cast<int>().All(x => x == 0))
        {
            return BaseFunction.BroadcastOutputNames(padCall, input);
        }

        return null;
    }
}

/// <summary>
/// Broadcast reshape outputs names.
/// </summary>
[RuleGenerator]
public sealed partial class BroadcastReshapeOutputNames : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape("reshape", "reshapeCall", IsWildcard("input"), IsWildcard("newShape"));

    private Expr? GetReplace(Call reshapeCall, Expr input, Expr newShape)
    {
        return BaseFunction.BroadcastOutputNames(reshapeCall, input);
    }
}

/// <summary>
/// Broadcast transpose outputs names.
/// </summary>
[RuleGenerator]
public sealed partial class BroadcastTransposeOutputNames : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose("tp", "tpCall", _ => true, IsWildcard("input"), IsTensorConst("perm"));

    private Expr? GetReplace(Call tpCall, Expr input, Tensor<int> perm)
    {
        return BaseFunction.BroadcastOutputNames(tpCall, input);
    }
}
