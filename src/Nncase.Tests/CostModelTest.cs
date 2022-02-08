using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.F;
using Xunit;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.Pattern.Utility;

namespace Nncase.Tests
{

    public class ExprCostModelTest
    {
        [Fact]
        public void TestConst()
        {
            var a = (Const)7;
            var exprVisitor = new ExprCostModelVisitor();
            Assert.True(a.InferenceType());
            Assert.Equal(new Cost(0, 4), exprVisitor.Visit(a));
        }

        [Fact]
        public void TestBinary()
        {
            // todo need process pow lhs is not cost
            // var a = (Const)1;
            // var n = (Const)5;
            // var pow = Math.Pow(a, n);
            // CompilerServices.InferenceType(pow);
            // var exprVisitor = new ExprCostModelVisitor();
            // Assert.Equal(new Cost(5, 0), exprVisitor.Visit(pow));
        }

        [Fact]
        public void TestCostInf()
        {
            var c = Cost.Inf;
            Assert.Equal(Cost.Inf, c + new Cost(10, 120));
        }
    }
}