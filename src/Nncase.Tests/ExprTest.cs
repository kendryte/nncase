using Xunit;
using Nncase;
using Nncase.IR;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Immutable;


public class UnitTestExpr
{
    [Fact]
    public void TestConstEqual()
    {
        var a = (Const)(1.1f) == (Const)(1.1f);
        Assert.True(a);
        var b = (Const)(1.1f) == (Const)(1.2f);
        Assert.False(b);
    }

    [Fact]
    public void TestConstEqualWithCheckType()
    {
        var a = (Const)(1.1f);
        var b = (Const)(1.1f);
        a.CheckedType = a.ValueType;
        Assert.True(a == b);
        var d = new HashSet<Const>();
        d.Add(a);
        Assert.Contains(b, d);
    }

    [Fact]
    public void TestBinaryAddEqualWithCheckType()
    {
        var a = (Const)(1.1f) + (Const)(1.1f);
        var b = (Const)(2) + (new Var("c"));
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
        var a = (Const)(1.1f) + (Const)(1.1f);
        var b = (Const)(2) - (new Var("c"));

        Assert.False(a.Target == b.Target);
    }

    [Fact]
    public void TestDenseTenorEqual()
    {
        var t = new DenseTensor<int>(new[] { 1, 2, 3, 4 });
        var con = Const.FromTensor<int>(t);
        var con1 = Const.FromTensor<int>(t);
        Assert.Equal(con, con1);
    }


    [Fact]
    public void TestConstToDenseTenor()
    {
        var con = Const.FromSpan<int>(new[] { 1, 2, 3, 4, 5 }, new[] { 5 });
        var t = con.ToTensor<int>();
        Assert.Equal(1, t[0]);
        Assert.Equal(2, t[1]);
        Assert.Equal(3, t[2]);
        Assert.Equal(4, t[3]);
        Assert.Equal(5, t[4]);
    }

    [Fact]
    public void TestDenseTensorLength()
    {
        var t = new DenseTensor<int>(new[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        Assert.Equal(4, t.Length);
        Assert.Equal(2, t.Dimensions[0]);
    }

}