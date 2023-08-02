// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.Passes.Rules.Lower.BroadcastMarkerHelper;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.Lower;

// e.g. matmul(reshape(marker(x))) -> matmul(marker(reshape(marker(x))))
[RuleGenerator]
public partial class BroadcastInputMarker : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCallWildcard(
        "outer",
        IsWildcard(),
        InputPattern);

    public Pattern InputPattern => IsCallWildcard(
        "call",
        IsWildcard(),
        IsRangeOfMarker(
            "marker",
            IsWildcard(),
            IsWildcard()));

    public Expr? GetReplace(Call outer, Call call, Marker marker)
    {
        if (!NotChangeRangeOp(call.Target))
        {
            return null;
        }

        if (outer.Target is MatMul && CompilerServices.TryMatchRoot(outer.Arguments[1], InputPattern, new(), out var matchResult))
        {
            var rhsMarker = (Marker)matchResult["marker"];
            var rhsCall = (Call)matchResult["call"];
            var lhs = marker.With(target: ReplaceCallFirstParam(call, marker));
            var rhs = rhsMarker.With(target: ReplaceCallFirstParam(rhsCall, rhsMarker));
            return ReplaceCallParams(outer, (0, lhs), (1, rhs));
        }

        return ReplaceCallFirstParam(outer, marker.With(target: ReplaceCallFirstParam(call, marker)));
    }
}

// e.g. marker(reshape(matmul(x))) -> marker(reshape(marker(matmul(x))))
[RuleGenerator]
public partial class BroadcastOutputMarker : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsRangeOfMarker(
        "marker",
        IsCallWildcard("input", IsWildcard(), IsCallWildcard(null, IsWildcard())),
        IsWildcard());

    public Expr? GetReplace(Call input, Marker marker)
    {
        if (!NotChangeRangeOp(input.Target))
        {
            return null;
        }

        return ReplaceCallFirstParam(input, marker.With(target: input.Arguments[0]));
    }
}

internal static class BroadcastMarkerHelper
{
    public static bool NotChangeRangeOp(Expr op)
    {
        return op is Squeeze || op is Unsqueeze || op is Reshape || op is Broadcast;
    }
}
