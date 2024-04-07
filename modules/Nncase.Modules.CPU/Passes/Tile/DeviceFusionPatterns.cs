// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Tile;

internal static class DeviceFusionPatterns
{
    public static Pattern UnaryUnaryPattern()
    {
        var v0 = IsVar("input");
        var v1 = PatternMatch.F.Math.IsUnary(null, "callee", _ => true, v0);
        var v2 = PatternMatch.F.Math.IsUnary(null, "caller", _ => true, v1);
        return v2;
    }

    public static Pattern MatmulUnaryPattern()
    {
        var v00 = IsVar("lhs");
        var v01 = IsVar("rhs");
        var v1 = PatternMatch.F.Math.IsMatMul(null, "callee", _ => true, v00, v01);
        var v2 = PatternMatch.F.Math.IsUnary(null, "caller", _ => true, v1);
        return v2;
    }
}
