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
        Assert.Equal(b, true);
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
        Assert.Equal(t[0],1);
        Assert.Equal(t[1],2);
        Assert.Equal(t[2],3);
        Assert.Equal(t[3],4);
        Assert.Equal(t[4],5);
    }

}