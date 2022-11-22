using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using static Nncase.IR.F.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Tensorflow;
using Xunit;
using ITuple = Nncase.IR.ITuple;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestCombineUnary : TestFixture.UnitTestFixtrue
{
    // TODO: CombinePadUnary
    public static IEnumerable<object[]> TestCombinePadUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] {1, 3, 4, 5},  new[,] {{ 1, 1 },
            { 2, 2 },
            { 1, 1 },
            { 3, 3 }}, PadMode.Symmetric, 0f},
            new object[] { UnaryOp.Abs, new[] {1, 3, 4, 5},  new[,] {{ 1, 1 },
            { -1, -1 },
            { 1, 1 },
            { 3, 3 }}, PadMode.Reflect, 0f},
            new object[] { UnaryOp.Floor, new[] {1, 3, 4, 5},  new[,] {{ 1, 1 },
            { 0, 0 },
            { 1, 1 },
            { 0, 0 }}, PadMode.Constant , 2f},
            new object[] { UnaryOp.Floor, new[] {1, 3, 4, 5},  new[,] {{ 1, 1 },
            { 0, 0 },
            { 1, 3 },
            { 6, 0 }}, PadMode.Edge , 2f},
        };

    [Theory]
    [MemberData(nameof(TestCombinePadUnaryPositiveData))]
    public void TestCombinePadUnaryPositive(UnaryOp opType, int[] inShape, int[,] paddings, PadMode padM, float padValue)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Pad(a, paddings, padM, padValue));
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombinePadUnary(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    public static IEnumerable<object[]> TestCombineSliceUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] {6},  new[] {0}, new[] {4}, new[] {0}, new[] {1}},
            new object[] { UnaryOp.Abs, new[] {4, 5},  new[] {0, 0}, new[] {2, 3}, new[] {0,1}, new[] {1, 2}},
            new object[] { UnaryOp.Sqrt, new[] {3, 4, 5},  new[] {0, 0, 1}, new[] {2, 3, 4}, new[] { 0, 1, 2}, new[] {1, 2, 3}},
            new object[] { UnaryOp.Square, new[] {3,2, 4, 5},  new[] {0, 0,1, 1}, new[] {-1, 2, 3, 4}, new[] {-4, 1, 2, 3 }, new[] {1, 2, 3, 2}},
        };

    [Theory]
    [MemberData(nameof(TestCombineSliceUnaryPositiveData))]
    public void TestCombineSliceUnaryPositive(UnaryOp opType, int[] inShape, int[] begins, int[] ends, int[] axes, int[] strides)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Slice(a, begins, ends, axes, strides));
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineSliceUnary(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    // TODO: CombineReshapeUnary
    public static IEnumerable<object[]> TestCombineReshapeUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] {6}, new[] {2,3}},
            new object[] { UnaryOp.Abs, new[] {4, 5}, new[] {2, 2, 5}, },
            new object[] { UnaryOp.Sqrt, new[] {3, 4, 5}, new[] {3, 2, 2, 5} },
            new object[] { UnaryOp.Square, new[] {3,2, 4, 5}, new[] {3,4, 1, 10}  },
        };

    [Theory]
    [MemberData(nameof(TestCombineReshapeUnaryPositiveData))]
    public void TestCombineReshapeUnaryPositive(UnaryOp opType, int[] inShape, int[] outShape)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Reshape(a, outShape));
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineReshapeUnary(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }
}