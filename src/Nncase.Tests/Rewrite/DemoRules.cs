using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.F;
using Xunit;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.Pattern.Utility;

namespace Nncase.Tests.ReWrite
{
    /// <summary>
    /// x * 1 => x
    /// </summary>
    public class DemoRuleXmul1 : PatternRule
    {
        CallPattern binary;
        public DemoRuleXmul1() => Pattern = binary = IsWildCard() * 1;
        public override Expr GetRePlace(IMatchResult result) => result[binary.Parameters[0]];
    }

    /// <summary>
    /// x / x => 1
    /// </summary>
    public class DemoRuleXDivX : PatternRule
    {
        WildCardPattern x = IsWildCard();
        public DemoRuleXDivX() => Pattern = x / x;
        public override Expr GetRePlace(IMatchResult result) => 1;
    }

    /// <summary>
    /// (x * y) / z  = x * (y / z)
    /// </summary>
    public class DemoRuleReassociateXYZ : PatternRule
    {
        WildCardPattern x = "x", y = "y", z = "z";
        public DemoRuleReassociateXYZ() => Pattern = (x * y) / z;
        public override Expr GetRePlace(IMatchResult result) => result[x] * (result[y] / result[z]);
    }

    /// <summary>
    /// (x * y) => y * x
    /// </summary>
    public class DemoRuleReassociateXY : PatternRule
    {
        WildCardPattern x = "x", y = "y";
        public DemoRuleReassociateXY() => Pattern = x * y;
        public override Expr GetRePlace(IMatchResult result) => result[y] * result[x];
    }
}