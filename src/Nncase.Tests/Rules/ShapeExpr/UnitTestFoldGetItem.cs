using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules.ShapeExpr;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.ShapeExpr;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldGetItem : TransformTestBase
{
    private readonly ITestOutputHelper _testOutputHelper;

    public UnitTestFoldGetItem(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
    }

    [Fact]
    public async Task TestFoldStackGetItem()
    {
        Expr input = new[] { 1, 2, 3, 4 };
        var s = Stack(new IR.Tuple(new[] { input[0], input[1], input[2], input[3] }), 0);
        TestMatched<FoldStackGetItem>(s);
    }

    [Fact]
    public async Task TestFoldSqueezeGetItem()
    {
        var shape = new Expr[] { 1, 80, 4, 1 };
        var s = Stack(new IR.Tuple(shape), 0);
        var result = Stack(new IR.Tuple(s[0], s[1], s[2]), 0);
        DumpScope.Current.DumpIR(result, "result");
        TestNotMatch<FoldStackGetItem>(result);
    }

    [Fact]
    public async Task TestFoldStackGetItemErrIndex()
    {
        Expr input = new[] { 1, 2, 3, 4 };
        var s = Stack(new IR.Tuple(new[] { input[0], input[1], input[3], input[2] }), 0);
        TestNotMatch<FoldStackGetItem>(s);
    }

    [Fact]
    public async Task TestFoldStackGetItemDiffInput()
    {
        Expr i0 = new[] { 1, 2, 3, 4 };
        Expr i1 = new[] { 5, 6, 7, 8 };
        Expr i2 = new[] { 9, 10, 11, 12 };
        var s = Stack(new IR.Tuple(new[] { i0, i1, i2 }), 0);
        TestNotMatch<FoldStackGetItem>(s);
    }
}
