// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Pattern;
using Nncase.Pattern.Math;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Tensors;
using static Nncase.Pattern.F.Tensors;
using Nncase.Pattern.Tensors;
using System.Numerics.Tensors;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.Transform.Rule
{
    public sealed class ReassociateMul : IRewriteRule
    {
        private WildcardPattern wx = "x", wy = "y", wz = "z";

        public ReassociateMul()
        {
            Pattern = (wx * wy) * wz;
        }

        public override Expr GetReplace(IMatchResult result)
        {
            var (x, y, z) = result[wx, wy, wz];
            return x * (y * z);
        }
    }

    /// <summary>
    /// x * 2 => x leftshift 1.
    /// </summary>
    public class Xmul2 : IRewriteRule
    {
        BinaryWrapper binary;
        public Xmul2() => Pattern = binary = IsBinary(BinaryOp.Mul, IsWildcard().SetTypePattern(IsIntegral() & IsScalar()), 2);
        public override Expr GetReplace(IMatchResult result)
        {
            binary.Bind(result);
            return binary.Lhs() << 1;
        }
    }

    /// <summary>
    /// x * 1 => x.
    /// </summary>
    public class Xmul1 : IRewriteRule
    {
        CallPattern binary;
        public Xmul1() => Pattern = binary = IsWildcard() * 1;
        public override Expr GetReplace(IMatchResult result) => result[binary.Parameters[0]];
    }

    /// <summary>
    /// x / x => 1.
    /// </summary>
    public class XDivX : IRewriteRule
    {
        WildcardPattern x = IsWildcard();
        public XDivX() => Pattern = x / x;
        public override Expr GetReplace(IMatchResult result) => 1;
    }

    /// <summary>
    /// (x * y) / z  = x * (y / z).
    /// </summary>
    public class ReassociateDiv : IRewriteRule
    {
        WildcardPattern x = "x", y = "y", z = "z";
        public ReassociateDiv() => Pattern = (x * y) / z;
        public override Expr GetReplace(IMatchResult result) => result[x] * (result[y] / result[z]);
    }

    /// <summary>
    /// (x * y) => y * x.
    /// </summary>
    public class ReassociateXY : IRewriteRule
    {
        WildcardPattern x = "x", y = "y";
        public ReassociateXY() => Pattern = x * y;
        public override Expr GetReplace(IMatchResult result) => result[y] * result[x];
    }
}