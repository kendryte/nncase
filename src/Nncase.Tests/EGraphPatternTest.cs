using System;
using Xunit;
using Nncase.Transform.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using PF = Nncase.Transform.Pattern.Functional;
using PM = Nncase.Transform.Pattern.Math;
using static Nncase.IR.F.Math;

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
            ExprPattern cp1 = con;
            Assert.IsType<ConstPattern>(cp1);

            ExprPattern cp2 = PF.IsConst((float x) => x > 1.2f);
            ExprPattern cp3 = PF.IsConst((int x) => x > 1);

            Assert.Equal(cp1.MatchLeaf(con), true);
            Assert.Equal(cp2.MatchLeaf(con), false);
            Assert.Equal(cp3.MatchLeaf(con), false);
        }

        [Fact]
        public void TestWildcardPattern()
        {
            var wc = PF.WildCard();
            Assert.IsType<WildCardPattern>(wc);
        }


        [Fact]
        public void TestCallPattern()
        {
            var e = (Const)1 + Exp(10);

            var wc1 = PF.WildCard();
            var wc2 = PF.WildCard();
            var c = wc1 + wc2;
            Assert.IsType<CallPattern>(c);
            Assert.IsType<PM.BinaryPattern>(c.Target);
            Assert.IsType<WildCardPattern>(c.Parameters[0]);
            Assert.IsType<WildCardPattern>(c.Parameters[1]);

            var c2 = PF.IsBinary(BinaryOp.Add, wc1, wc2);

            var c3 = PF.IsBinary(x => x is (BinaryOp.Div or BinaryOp.Sub), wc1, wc2);

            Assert.Equal(c.Target.MatchLeaf(e.Target), true);
            Assert.Equal(c2.Target.MatchLeaf(e.Target), true);
            Assert.Equal(c3.Target.MatchLeaf(e.Target), false);
        }

        [Fact]
        public void TestFunctionPattern()
        {
            var wc1 = PF.WildCard();
            var wc2 = PF.WildCard();
            var c = wc1 + wc2;
            var fp = new FunctionPattern(new[] { wc1, wc2 }, c);
            Assert.IsType<FunctionPattern>(fp);
            Assert.IsType<WildCardPattern>(fp.Parameters[0]);
            Assert.IsType<WildCardPattern>(fp.Parameters[1]);
            Assert.IsType<CallPattern>(fp.Body);
            Assert.IsType<WildCardPattern>(((CallPattern)fp.Body).Parameters[0]);
            Assert.IsType<WildCardPattern>(((CallPattern)fp.Body).Parameters[1]);
        }

        [Fact]
        public void TestTuplePattern()
        {
            var wc1 = PF.WildCard();
            var wc2 = PF.WildCard();
            var t = PF.IsTuple(new[] { wc1, wc2 });
            Assert.IsType<TuplePattern>(t);
            Assert.IsType<WildCardPattern>(t.Fields[0]);
            Assert.IsType<WildCardPattern>(t.Fields[1]);
        }

        [Fact]
        public void TestAltPattern()
        {
            var lhs = PF.WildCard();
            var rhs = PF.WildCard();
            var is_op_call = new CallPattern(PF.WildCard(), lhs, rhs);
            Const x = (Const)1;
            Const y = (Const)2;
            var z1 = x + y;
            var z2 = x * y;
            Assert.Equal(is_op_call.MatchLeaf(z1), true);
            Assert.Equal(is_op_call.Target.MatchLeaf(z2.Target), true);
        }

        [Fact]
        public void TestTypePattern()
        {
            var ttype = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ty_pat = PF.HasType(ttype);
            Assert.IsType<TypePattern>(ty_pat);
            Assert.Equal(ty_pat.MatchLeaf(ttype), true);
        }

        [Fact]
        public void TestDataTypePattern()
        {
            var ttype1 = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ttype2 = new TensorType(DataType.Int16, new[] { 10 });
            var ty_pat = PF.HasDType(DataType.Float32);
            Assert.IsType<TypePattern>(ty_pat);
            Assert.Equal(ty_pat.MatchLeaf(ttype1), true);
            Assert.Equal(ty_pat.MatchLeaf(ttype2), false);
        }

        [Fact]
        public void TestShapePattern()
        {
            var shape = new int[] { 10, 10 };
            var sp = PF.HasShape(shape);
            var ttype1 = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ttype2 = new TensorType(DataType.Int16, new[] { 10 });
            Assert.Equal(sp.MatchLeaf(ttype1), true);
            Assert.Equal(sp.MatchLeaf(ttype2), false);
        }
    }

    public class UnitTestGraphMatch
    {
        [Fact]
        public void TestMatchOpAlt()
        {
            var wc1 = PF.WildCard();
            var pat = wc1 + 1;

            var a = new Var("a");
            var e = a * 100 - 32 / 320 + 1;
            var g = new EGraph();
            g.Add(e);

            var matched = EClassMatcher.EMatch(g, pat);
        }
    }
}
