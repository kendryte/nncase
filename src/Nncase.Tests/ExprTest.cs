using Xunit;
using Nncase.IR;

public class UnitTestExpr
{
    [Fact]
    public void TestConstEqual()
    {
        var b = (Const)(1.1f) == (Const)(1.1f);
        Assert.Equal(b, true);
    }
}