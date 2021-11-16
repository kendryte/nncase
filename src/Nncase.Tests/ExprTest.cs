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
    public void TestOPEqualWithCheckType()
    {
        var a = (Const)(1.1f) + (Const)(1.1f);
        var b = (Const)(2) + (new Var("c"));

        Assert.True(a.Target == b.Target);

        var paramTypes = ((Op)a.Target).Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
        var type = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypes));
        a.Target.CheckedType = type;

        Assert.True(a.Target.Equals(b.Target));
        Assert.True(a.Target == b.Target);
        var d = new HashSet<Expr>();
        d.Add(a.Target);
        Assert.Contains(b.Target, d);

        var paramTypesb = ((Op)b.Target).Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
        var typeb = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypesb));
        b.Target.CheckedType = typeb;

        Assert.True(a.Target == b.Target);
        Assert.Contains(b.Target, d);
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