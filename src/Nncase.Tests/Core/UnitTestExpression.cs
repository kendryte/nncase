// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public class UnitTestExpression
{
    [Fact]
    public void TestConstEqual()
    {
        var a = (Const)1.1f == (Const)1.1f;
        Assert.True(a);
        var b = (Const)1.1f == (Const)1.2f;
        Assert.False(b);

        var va = (Const)new[] { 1, 2, 3, 4 };
        var vb = (Const)new[] { 1, 2, 3, 4 };
        Assert.Equal(va, vb);
        Assert.Equal(va.GetHashCode(), vb.GetHashCode());

        var sa = new TensorType(DataTypes.Int32, new Shape(new[] { 2 }));
        var sb = new TensorType(DataTypes.Int32, new Shape(new[] { 2 }));
        Assert.True(sa.Shape == sb.Shape);
        Assert.True(sa == sb);
        Assert.Equal(sa, sb);
        Assert.Equal(sa.GetHashCode(), sb.GetHashCode());
    }

    /// <summary>
    /// when check type is different, expression not equal.
    /// </summary>
    [Fact]
    public void TestConstEqualWithCheckType()
    {
        var a = (Const)1.1f;
        var b = (Const)1.1f;
        a.CheckedType = a.ValueType;
        Assert.True(a == b);
        Assert.Equal(a, b);
        var d = new HashSet<Const>();
        d.Add(a);
        Assert.Contains(b, d);
    }

    [Fact]
    public void TestCallEqualWithCheckType()
    {
        var a = (Const)1.1f + (Const)1.3f;
        var b = (Const)1.1f + (Const)1.3f;
        CompilerServices.InferenceType(a);
        Assert.True(a == b);
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void TestCallNotEqualWithCheckType()
    {
        var a = (Const)1.1f + (Const)1.3f;
        var b = (Const)1.1f + (Const)1.2f;
        CompilerServices.InferenceType(a);
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void TestTupleGetHashCode()
    {
        var expr = new IR.Tuple((Const)1 * (Const)2, (Const)1.0f + (Const)2.4f);
        var d = new HashSet<Expr>() { (Const)1.3f };
        d.TryGetValue(expr, out _);
    }

    [Fact]
    public void TestTupleGetHash()
    {
        var a = new IR.Tuple((Const)1 * (Const)2);
        var b = new IR.Tuple(a, a, a, a);
        var c = new IR.Tuple(b, b, b, b);
        var d = new IR.Tuple(c, c, c, c);
        var expr = new IR.Tuple(d, d, d, d);
        var set = new HashSet<Expr>();
        set.Add(expr);
    }

    [Fact]
    public void TestTupleGetHashDifference()
    {
        Expr a = (Const)1;
        Expr b = (Const)3;
        Assert.NotEqual(a, b);
        int ahash1 = a.GetHashCode();
        int ahash2 = ((Const)a).GetHashCode();
        Assert.Equal(ahash1, ahash2);
        _ = new HashSet<Expr>();
    }

    [Fact]
    public void TestBinaryAddEqualWithCheckType()
    {
        var a = (Const)1.1f + (Const)1.1f;
        var b = (Const)2 + new Var("c");
        var dict = new Dictionary<Expr, int> { };
        Op opa = (Op)a.Target, opb = (Op)b.Target;

        Assert.True(opa == opb);
        dict.Add(opa, 0);
        Assert.True(dict.ContainsKey(opa));

        var paramTypes = opa.Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
        var type = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypes));
        opa.CheckedType = type;

        Assert.Contains(opa, dict.Keys);
        Assert.True(dict.ContainsKey(opa));
        Assert.True(opa.Equals(opb));
        Assert.True(opa == opb);

        Assert.True(dict.TryGetValue(opb, out var result));

        var paramTypesb = opb.Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
        var typeb = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypesb));
        opb.CheckedType = typeb;

        Assert.True(opa == opb);
        Assert.Contains(opb, dict.Keys);
    }

    [Fact]
    public void TestBinaryOpEqualWithCheckType()
    {
        var a = (Const)1.1f + (Const)1.1f;
        var b = (Const)2 - new Var("c");

        Assert.False(a.Target == b.Target);

        var c = (Const)1.1f + (Const)1.1f;

        Assert.True(a.Equals(c));
    }

    [Fact]
    public void TestDenseTenorEqual()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3, 4 });
        var con = Const.FromTensor(t);
        var con1 = Const.FromTensor(t);
        Assert.Equal(con, con1);
        Assert.False(object.ReferenceEquals(con, con1));
    }

    [Fact]
    public void TestConstToDenseTenor()
    {
        var con = Const.FromTensor(Tensor.From<int>(new[] { 1, 2, 3, 4, 5 }, new[] { 5 }));
        var t = con.Value.Cast<int>();
        Assert.Equal(1, t[0]);
        Assert.Equal(2, t[1]);
        Assert.Equal(3, t[2]);
        Assert.Equal(4, t[3]);
        Assert.Equal(5, t[4]);
        var t2 = con.Value.Cast<long>();
        Assert.Equal(1, t2[0]);
        Assert.Equal(2, t2[1]);
        Assert.Equal(3, t2[2]);
        Assert.Equal(4, t2[3]);
        Assert.Equal(5, t2[4]);
        _ = con.Value.Cast<byte>();
        Assert.Equal(1, t2[0]);
        Assert.Equal(2, t2[1]);
        Assert.Equal(3, t2[2]);
        Assert.Equal(4, t2[3]);
        Assert.Equal(5, t2[4]);
        _ = con.Value.Cast<float>();
        Assert.Equal(1.0f, t2[0]);
        Assert.Equal(2.0f, t2[1]);
        Assert.Equal(3.0f, t2[2]);
        Assert.Equal(4.0f, t2[3]);
        Assert.Equal(5.0f, t2[4]);
    }

    [Fact]
    public void TestDenseTensorLength()
    {
        var t = new Tensor<int>(new[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        Assert.Equal(4, t.Length);
        Assert.Equal(2, t.Dimensions[0]);
    }

    [Fact]
    public void TestHastSet()
    {
        var a_lhs = (Const)1.1f;
        var a_rhs = (Const)1.1f;
        var a = a_lhs + a_rhs;
        var b_lhs = (Const)1.1f;
        var b_rhs = new Var("c");
        var b = b_lhs - b_rhs;
        var c = a_lhs + a_rhs;
        var set = new HashSet<Expr>()
        {
            a_lhs,
            a_rhs, // will fold
            a,
            b_lhs, // will fold
            b_rhs,
            b,
            c, // will fold
        };

        Assert.Equal(4, set.Count);
    }

    [Fact]
    public void TestShapeGetHash()
    {
        var n = new Dimension(1);
        var c = new Dimension(1);
        Assert.StrictEqual(c, n);

        var a = ImmutableArray.CreateRange(new[] { n, c });
        var b = ImmutableArray.CreateRange(new[] { new Dimension(1), new Dimension(1) });
        Assert.NotEqual(a, b);

        var sa = new Shape(new[] { 1, 2, 3 });
        var sb = new Shape(new[] { 1, 2, 3 });
        Assert.StrictEqual(sa, sb);
        Assert.Equal(sa, sb);
    }

    [Fact]
    public void TestOpTypeGetHash()
    {
        var a = new IR.Math.Binary(BinaryOp.Add);
        var b = new IR.Math.Binary(BinaryOp.Add);
        var c = new IR.Math.Unary(UnaryOp.Acos);
        Assert.Equal(a, b);

        Dictionary<Op, Evaluator.IEvaluator> dict1 = new();
        Dictionary<Type, Evaluator.IEvaluator> dict2 = new();
        Dictionary<RuntimeTypeHandle, Evaluator.IEvaluator> dict3 = new();
        for (int i = 0; i < 1000000; i++)
        {
            // method 1
            if (!dict1.TryGetValue(a, out _))
            {
                Evaluator.IEvaluator? d1_a = new Evaluator.Math.BinaryEvaluator();
                dict1.Add(a, d1_a);
            }

            if (!dict1.TryGetValue(b, out _))
            {
                Evaluator.IEvaluator? d1_b = new Evaluator.Math.BinaryEvaluator();
                dict1.Add(b, d1_b);
            }

            if (!dict1.TryGetValue(c, out _))
            {
                Evaluator.IEvaluator? d1_c = new Evaluator.Math.UnaryEvaluator();
                dict1.Add(c, d1_c);
            }

            // method 2
            if (!dict2.TryGetValue(a.GetType(), out _))
            {
                Evaluator.IEvaluator? d2_a = new Evaluator.Math.BinaryEvaluator();
                dict2.Add(a.GetType(), d2_a);
            }

            if (!dict2.TryGetValue(b.GetType(), out _))
            {
                Evaluator.IEvaluator? d2_b = new Evaluator.Math.BinaryEvaluator();
                dict2.Add(b.GetType(), d2_b);
            }

            if (!dict2.TryGetValue(c.GetType(), out _))
            {
                Evaluator.IEvaluator? d2_c = new Evaluator.Math.UnaryEvaluator();
                dict2.Add(c.GetType(), d2_c);
            }

            // method 3
            if (!dict3.TryGetValue(a.GetType().TypeHandle, out _))
            {
                Evaluator.IEvaluator? d3_a = new Evaluator.Math.BinaryEvaluator();
                dict3.Add(a.GetType().TypeHandle, d3_a);
            }

            if (!dict3.TryGetValue(b.GetType().TypeHandle, out _))
            {
                Evaluator.IEvaluator? d3_b = new Evaluator.Math.BinaryEvaluator();
                dict3.Add(b.GetType().TypeHandle, d3_b);
            }

            if (!dict3.TryGetValue(c.GetType().TypeHandle, out _))
            {
                Evaluator.IEvaluator? d3_c = new Evaluator.Math.UnaryEvaluator();
                dict3.Add(c.GetType().TypeHandle, d3_c);
            }
        }
    }

    [Fact]
    public void TestPrintExpr()
    {
        Expr x = new int[] { 1, 2, 3, 4 };
        CompilerServices.InferenceType(x);
        Assert.Equal("const(i32[4] : {1,2,3,4})", CompilerServices.Print(x));
        Assert.Equal("None", CompilerServices.Print(None.Default));
        Assert.Equal("Add", CompilerServices.Print(new Nncase.IR.Math.Binary(BinaryOp.Add)));
        var y = new Var("y");
        CompilerServices.InferenceType(y);
        Assert.Equal("%y: any", CompilerServices.Print(y));
    }

    [Fact]
    public void TestExpressionTree()
    {
        var input_1 = new Var("input_1", TensorType.Scalar(DataTypes.Int32));
        var fn_1 = new Function("add", IR.F.Math.Binary(BinaryOp.Add, input_1, 10), new[] { input_1 });
        Assert.True(CompilerServices.InferenceType(fn_1));

        var visitor = new ExpressionTreeBuilder();
        var fn_2 = ((LambdaExpression)visitor.Visit(fn_1)).Compile();

        for (int i = 0; i < 100000; i++)
        {
            var res_1 = CompilerServices.Evaluate(fn_1.Body, new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance) { { input_1, Value.FromConst(i) } }).AsTensor().ToScalar<int>();
            Assert.Equal(i + 10, res_1);
            Assert.Equal(res_1, fn_2.DynamicInvoke(i));
        }
    }

    private sealed class ExpressionTreeBuilder : ExprVisitor<Expression, Type>
    {
        public override Expression VisitLeaf(Const expr)
        {
            if (expr is TensorConst tc && tc.Value.Shape.IsScalar)
            {
                return Expression.Constant(tc.Value[0], tc.Value.ElementType.CLRType);
            }

            throw new ArgumentOutOfRangeException();
        }

        public override Expression VisitLeaf(Var expr)
        {
            if (expr.CheckedShape.IsScalar)
            {
                return Expression.Parameter(expr.CheckedDataType.CLRType, expr.Name);
            }

            throw new ArgumentOutOfRangeException();
        }

        public override Expression VisitLeaf(Call expr)
        {
            switch (expr.Target)
            {
                case IR.Math.Binary binary:

                    return binary.BinaryOp switch
                    {
                        BinaryOp.Add => Expression.Add(Visit(expr.Parameters[0]), Visit(expr.Parameters[1])),
                        _ => throw new ArgumentOutOfRangeException(),
                    };
                default:
                    break;
            }

            throw new ArgumentOutOfRangeException();
        }

        public override Expression VisitLeaf(Function expr)
        {
            return Expression.Lambda(Visit(expr.Body), expr.Name, expr.Parameters.Select(v => (ParameterExpression)Visit(v)).ToArray());
        }

        public override Expression VisitLeaf(Op expr)
        {
            return null!;
        }
    }

    [Fact]
    public void TestExpressionTreeWithCustomStruct()
    {
        var root_range = Expression.Parameter(typeof(MyRange[]), "root_range");
        Expression<Func<MyRange[], MyRange[]>> conv2d_1 = (MyRange[] output_range) => Conv2dBounds(output_range, 3, 4);
        Expression<Func<MyRange[], MyRange[]>> conv2d_2 = (MyRange[] output_range) => Conv2dBounds(output_range, 5, 6);

        var body_1 = Expression.Invoke(conv2d_1, root_range);
        var fn_1 = Expression.Lambda<Func<MyRange[], MyRange[]>>(body_1, root_range).Compile();
        var fn_1_ret = fn_1(new MyRange[] { new(1, 2), new(3, 4), new(5, 6), new(7, 9) });
        Assert.Equal<MyRange>(new(5 + 3, 6 + 3), fn_1_ret[2]);
        Assert.Equal<MyRange>(new(7 + 4, 9 + 4), fn_1_ret[3]);

        var body_2 = Expression.Invoke(conv2d_2, body_1);
        var fn_2 = Expression.Lambda<Func<MyRange[], MyRange[]>>(body_2, root_range).Compile();
        var fn_2_ret = fn_2(new MyRange[] { new(1, 2), new(3, 4), new(5, 6), new(7, 9) });
        Assert.Equal<MyRange>(new(5 + 3 + 5, 6 + 3 + 5), fn_2_ret[2]);
        Assert.Equal<MyRange>(new(7 + 4 + 6, 9 + 4 + 6), fn_2_ret[3]);
    }

    private MyRange[] Conv2dBounds(MyRange[] output_range, int kh, int kw)
    {
        var new_output_range = new MyRange[output_range.Length];
        Array.Copy(output_range, new_output_range, output_range.Length);
        new_output_range[2].Start += kh;
        new_output_range[2].Stop += kh;
        new_output_range[3].Start += kw;
        new_output_range[3].Stop += kw;
        return new_output_range;
    }

    private struct MyRange
    {
        public int Start;
        public int Stop;

        public MyRange(int start, int stop)
        {
            Start = start;
            Stop = stop;
        }
    }
}
