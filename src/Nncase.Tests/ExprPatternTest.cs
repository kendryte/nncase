using System;
using Xunit;
using Nncase.Pattern;
using Nncase.Transform;
using Nncase.IR;
using System.Collections.Generic;
using Nncase.Pattern.Math;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.IR.Utility;

namespace Nncase.Tests
{

    public class UnitTestExprPattern
    {

        [Fact]
        public void TestPatternEqual()
        {
            ExprPattern p1 = IsWildCard();
            int p1hash = p1.GetHashCode();
            ExprPattern p2 = IsBinary(IsConst(), IsConst());
            Assert.NotEqual(p1, p2);
            ExprPattern p3 = p1.IsScalar();
            int p3hash = p3.GetHashCode();
            Assert.Equal(p1hash, p3hash);

            OpPattern op1 = new UnaryPattern();
            OpPattern op2 = new BinaryPattern();
            OpPattern op3 = new BinaryPattern();
            Assert.NotEqual(op1, op2);
            Assert.NotEqual(op1.GetHashCode(), op2.GetHashCode());
            Assert.NotEqual(op2, op3);
            Assert.NotEqual(op2.GetHashCode(), op3.GetHashCode());
        }

        [Fact]
        public void TestVarPattern()
        {
            Var e = new Var("x", AnyType.Default);
            ExprPattern ep = e;
            Assert.IsType<VarPattern>(ep);
            Assert.True(ep.MatchLeaf(e));
        }

        [Fact]
        public void TestConstantPattern()
        {
            var con = (Const)(1.1f);
            ExprPattern cp1 = con;
            Assert.IsType<ConstPattern>(cp1);

            ExprPattern cp2 = IsConst((float x) => x > 1.2f);
            ExprPattern cp3 = IsConst((int x) => x > 1);
            var cp4 = (ConstPattern)1.1f;

            Assert.True(cp1.MatchLeaf(con));
            Assert.False(cp2.MatchLeaf(con));
            Assert.False(cp3.MatchLeaf(con));
            Assert.True(cp4.MatchLeaf(con));
        }

        [Fact]
        public void TestConstantPatternEqual()
        {
            ConstPattern cp1 = (ConstPattern)1;
            ConstPattern cp2 = (ConstPattern)1;
            Dictionary<ConstPattern, int> d = new();
            d.Add(cp1, 1);
            Assert.NotEqual(cp1, cp2);
            Assert.DoesNotContain(cp2, d.Keys);
            ConstPattern cp3 = IsConst();
            ConstPattern cp4 = IsConst();
            d.Add(cp3, 1);
            Assert.NotEqual(cp3, cp4);
            Assert.DoesNotContain(cp4, d.Keys);
        }

        [Fact]
        public void TestWildcardPattern()
        {
            var wc = IsWildCard();
            Assert.IsType<WildCardPattern>(wc);
        }

        [Fact]
        public void TestWildCardPatternHash()
        {
            var wc = IsWildCard();
            var wc2 = new WildCardPattern();
            var wc3 = new WildCardPattern(wc2.Name);
            var d = new Dictionary<WildCardPattern, int>();
            d.Add(wc, 1);
            d.Add(wc2, 2);
            d.Add(wc3, 2);
        }

        [Fact]
        public void TestCallPattern()
        {
            var e = (Const)1 + Exp(10);

            var wc1 = IsWildCard();
            var wc2 = IsWildCard();
            var c = wc1 + wc2;
            Assert.IsType<CallPattern>(c);
            Assert.IsType<BinaryPattern>(c.Target);
            Assert.IsType<WildCardPattern>(c.Parameters[0]);
            Assert.IsType<WildCardPattern>(c.Parameters[1]);

            CallPattern c2 = IsBinary(BinaryOp.Add, wc1, wc2);

            CallPattern c3 = IsBinary(x => x is (BinaryOp.Div or BinaryOp.Sub), wc1, wc2);

            Assert.True(c.Target.MatchLeaf(e.Target));
            Assert.True(c2.Target.MatchLeaf(e.Target));
            Assert.False(c3.Target.MatchLeaf(e.Target));
        }

        [Fact]
        public void TestFunctionPattern()
        {
            var wc1 = IsWildCard();
            var wc2 = IsWildCard();
            var c = wc1 + wc2;
            var fp = new FunctionPattern(c, wc1, wc2);
            Assert.IsType<FunctionPattern>(fp);
            Assert.IsType<WildCardPattern>(fp.Parameters[0]);
            Assert.IsType<WildCardPattern>(fp.Parameters[1]);
            Assert.IsType<CallPattern>(fp.Body);
            Assert.IsType<WildCardPattern>(((CallPattern)fp.Body).Parameters[0]);
            Assert.IsType<WildCardPattern>(((CallPattern)fp.Body).Parameters[1]);

            var fp2 = new FunctionPattern(c, IsVArgs(wc1, wc2));
            Assert.IsType<WildCardPattern>(fp.Parameters[0]);
            Assert.IsType<WildCardPattern>(fp.Parameters[1]);
        }

        [Fact]
        public void TestTuplePattern()
        {
            var wc1 = IsWildCard();
            var wc2 = IsWildCard();
            var t = IsTuple(wc1, wc2);
            Assert.IsType<TuplePattern>(t);
            Assert.IsType<WildCardPattern>(t.Fields[0]);
            Assert.IsType<WildCardPattern>(t.Fields[1]);

            var t2 = IsTuple(IsVArgs(wc1, wc2));
            Assert.IsType<TuplePattern>(t2);
            Assert.IsType<WildCardPattern>(t2.Fields[0]);
            Assert.IsType<WildCardPattern>(t2.Fields[1]);
        }

        [Fact]
        public void TestVArgsPattern()
        {
            var wc = IsWildCard();
            var vwcs = new List<ExprPattern>();
            var pattern = IsVArgsRepeat((n, param) =>
            {
                for (int i = 0; i < n; i++)
                {
                    var wc = IsWildCard();
                    param.Add(wc);
                    vwcs.Add(wc);
                }
            },
            (match, param) =>
            {
                if (match == false)
                {
                    param.Clear();
                    vwcs.Clear();
                }
            }
            );

            var tuple = new IR.Tuple(1, new IR.Tuple(6, 7, 8), 3, 4);
            Assert.True(pattern.MatchLeaf(tuple.Fields));
            Assert.True(pattern[0].MatchLeaf(tuple[0]));
            Assert.True(pattern[1].MatchLeaf(tuple[1]));
            Assert.True(pattern[2].MatchLeaf(tuple[2]));
            Assert.True(pattern[3].MatchLeaf(tuple[3]));
        }

        [Fact]
        public void TestVArgsPatternFunc()
        {
            var pat = IsVArgsRepeat(() => IsConst());
            IR.Tuple expr = new IR.Tuple(1, 2, 3, 4, 5, 6);
            pat.MatchLeaf(expr.Fields);
            Assert.Equal(pat.Count, expr.Fields.Count);
        }

        [Fact]
        public void TestAltPattern()
        {
            var lhs = IsWildCard();
            var rhs = IsWildCard();
            var is_op_call = IsCall(IsWildCard(), lhs, rhs);
            Const x = (Const)1;
            Const y = (Const)2;
            var z1 = x + y;
            var z2 = x * y;
            Assert.True(is_op_call.MatchLeaf(z1));
            Assert.True(is_op_call.Target.MatchLeaf(z2.Target));

            var is_op_call2 = IsCall(IsWildCard(), IsVArgs(lhs, rhs));

            Assert.IsType<WildCardPattern>(is_op_call2.Parameters[0]);
            Assert.IsType<WildCardPattern>(is_op_call2.Parameters[1]);
        }

        [Fact]
        public void TestTypePattern()
        {
            var ttype = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ty_pat = HasType(ttype);
            Assert.IsType<TypePattern>(ty_pat);
            Assert.True(ty_pat.MatchLeaf(ttype));
        }

        [Fact]
        public void TestDataTypePattern()
        {
            var ttype1 = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ttype2 = new TensorType(DataType.Int16, new[] { 10 });
            var ty_pat = HasDType(DataType.Float32);
            Assert.IsType<TypePattern>(ty_pat);
            Assert.True(ty_pat.MatchLeaf(ttype1));
            Assert.False(ty_pat.MatchLeaf(ttype2));
        }

        [Fact]
        public void TestShapePattern()
        {
            var shape = new int[] { 10, 10 };
            var sp = HasShape(shape);
            var ttype1 = new TensorType(DataType.Float32, new[] { 10, 10 });
            var ttype2 = new TensorType(DataType.Int16, new[] { 10 });
            Assert.True(sp.MatchLeaf(ttype1));
            Assert.False(sp.MatchLeaf(ttype2));
        }

        [Fact]
        public void TestPatternClone()
        {
            var pat = IsWildCard();
            var pat2 = IsWildCard().Copy();
            Assert.NotEqual(pat, pat2);
        }
    }
}