using System;
using Xunit;
using Nncase.Transform.Pattern;
using Nncase.IR;
using System.Collections.Generic;
using PF = Nncase.Transform.Pattern.Functional;

namespace Nncase.Tests
{
    public class UnitTestEGraphPattern
    {


        [Fact]
        public void TestVarPattern()
        {
            Var e = new Var("x", AnyType.Default);
            ExprPattern ep = e;
            Assert.IsType<VarPattern>(ep);
            Assert.Equal(ep.MatchLeaf(e), true);
        }

        [Fact]
        public void TestConstantPattern()
        {
            var con = (Const)(1.1f);
            ExprPattern cp = con;
            Assert.IsType<ConstPattern>(cp);
            Assert.Equal(cp.MatchLeaf(con), true);

            ExprPattern cp2 = PF.IsConst((float x) => x > 1.2f);
            
            ExprPattern cp3 = PF.IsConst((int x) => x > 1);

        }

        [Fact]
        public void TestWildcardPattern() { }
        [Fact]
        public void TestCallPattern() { }
        [Fact]
        public void TestFunctionPattern() { }
        [Fact]
        public void TestTuplePattern() { }
        [Fact]
        public void TestTupleGetItemPattern() { }
        [Fact]
        public void TestAltPattern() { }
        [Fact]
        public void TestTypePattern() { }
        [Fact]
        public void TestDataTypePattern() { }
        [Fact]
        public void TestShapePattern() { }
        [Fact]
        public void TestAttrPattern() { }
        [Fact]
        public void TestIfPattern() { }
        [Fact]
        public void TestLetPattern() { }
    }
}
