using Nncase.IR;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.ShapeExpr;

namespace Nncase.Tests.Rules.ShapeExpr;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldBroadcastShape : TransformTestBase
{
    [Fact]
    public void TestFoldBroadcastShape()
    {
        var b1 = BroadcastShape(new[] { (Expr)Tensor.From(new[] { 1, 3 }), Tensor.From(new[] { 1 }) });
        var b2 = BroadcastShape(new[] { (Expr)b1, Tensor.From(new[] { 1, 1 }) });
        TestMatched<FoldBroadcastShape>(b2);
    }
}
