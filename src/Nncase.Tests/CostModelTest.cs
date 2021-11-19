using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.F;
using Xunit;

namespace Nncase.Tests
{
    public class ExprCostModelTest
    {
        [Fact]
        public void TestConst()
        {
            var a = (Const) 7;
            var exprVisitor = new ExprCostModelVisitor();
            Assert.Equal(new Cost(), exprVisitor.Visit(a));
        }

        [Fact]
        public void TestBinary()
        {
            var a = (Const) 1;
            var n = (Const) 5;
            var pow = Math.Pow(a, n);
            TypeInference.InferenceType(pow);
            var exprVisitor = new ExprCostModelVisitor();
            Assert.Equal(new Cost(5, 0), exprVisitor.Visit(pow));
        }
    }

    public class EGraphCostModelTest
    {
        [Fact]
        public void TestConst()
        {
            
        }
    }
}