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
    public sealed class ReassociateMul : PatternRule
    {
        private WildCardPattern wx = "x", wy = "y", wz = "z";

        public ReassociateMul()
        {
            Pattern = (wx * wy) * wz;
        }

        public override Expr GetRePlace(IMatchResult result)
        {
            var (x, y, z) = result[wx, wy, wz];
            return x * (y * z);
        }
    }

    /// <summary>
    /// x * 2 => x leftshift 1
    /// </summary>
    public class Xmul2 : PatternRule
    {
        BinaryWrapper binary;
        public Xmul2() => Pattern = binary = IsBinary(BinaryOp.Mul, IsWildCard().SetTypePattern(IsIntegral() & IsScalar()), 2);
        public override Expr GetRePlace(IMatchResult result)
        {
            binary.Bind(result);
            return binary.Lhs() << 1;
        }
    }

    /// <summary>
    /// x * 1 => x
    /// </summary>
    public class Xmul1 : PatternRule
    {
        CallPattern binary;
        public Xmul1() => Pattern = binary = IsWildCard() * 1;
        public override Expr GetRePlace(IMatchResult result) => result[binary.Parameters[0]];
    }

    /// <summary>
    /// x / x => 1
    /// </summary>
    public class XDivX : PatternRule
    {
        WildCardPattern x = IsWildCard();
        public XDivX() => Pattern = x / x;
        public override Expr GetRePlace(IMatchResult result) => 1;
    }

    /// <summary>
    /// (x * y) / z  = x * (y / z)
    /// </summary>
    public class ReassociateDiv : PatternRule
    {
        WildCardPattern x = "x", y = "y", z = "z";
        public ReassociateDiv() => Pattern = (x * y) / z;
        public override Expr GetRePlace(IMatchResult result) => result[x] * (result[y] / result[z]);
    }

    /// <summary>
    /// (x * y) => y * x
    /// </summary>
    public class ReassociateXY : PatternRule
    {
        WildCardPattern x = "x", y = "y";
        public ReassociateXY() => Pattern = x * y;
        public override Expr GetRePlace(IMatchResult result) => result[y] * result[x];
    }

}