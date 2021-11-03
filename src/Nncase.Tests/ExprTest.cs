using Xunit;
using Nncase;
using Nncase.IR;
using System.Numerics.Tensors;

public class UnitTestExpr
{
    [Fact]
    public void TestConstEqual()
    {
        var b = (Const)(1.1f) == (Const)(1.1f);
        Assert.True(b);
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

}