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
using Microsoft.Toolkit.HighPerformance;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestCombineTranspose : TestFixture.UnitTestFixtrue
{
    // public static IEnumerable<object[]> TestCombineTransposeBinaryPositiveData =>
    //     new[]
    //     {
    //         new object[] {new[] {4}, new[] {4}, new[] {0}},
    //         new object[] {new[] {4, 4}, new[] {4, 4}, new[] {1, 0}},
    //         new object[] {new[] {1, 3, 4}, new[] {1, 3, 4}, new[] {0, 2, 1}},
    //         new object[] {new[] {1, 3, 2, 4}, new[] {1, 3, 2, 4}, new[] {0, 2, 3, 1}},
    //         // new object[] {new[] {5, 4, }, new[] {4, 6}, new[] {1, 0}},
    //     };
    //
    // [Theory]
    // [MemberData(nameof(TestCombineTransposeBinaryPositiveData))]
    // public void TestCombineTransposeBinaryPositive(int[] lShape, int[] rShape, int[] perm)
    // {
    //     var caseOptions = GetPassOptions();
    //     var a = new Var();
    //     var b = new Var();
    //
    //     var Normal = new Dictionary<Var, IValue>();
    //
    //     Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, lShape).Evaluate());
    //     Normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate());
    //
    //     // TODO: Rely cse to fold same constant. 
    //     var rootPre = Math.Binary(BinaryOp.Add, Tensors.Transpose(a, perm), Tensors.Transpose(b, perm));
    //     var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
    //     {
    //         // new FoldConstCall(),
    //         new CombineTransposeBinary(),
    //     }, caseOptions);
    //
    //     Assert.NotEqual(rootPre, rootPost);
    //     Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    // }

    // public static IEnumerable<object[]> TestCombineTransposeConcatPositiveData =>
    //     new[]
    //     {
    //         // new object[] {new[] {4, 4}, new[] {1, 0}, 1, 2},
    //         new object[] {new[] {1, 3, 4}, new[] {0, 2, 1}, 1, 6},
    //         // new object[] {new[] {1, 3, 2, 4}, new[] {0, 2, 3, 1}, 2, 1},
    //     };
    //
    // [Theory]
    // [MemberData(nameof(TestCombineTransposeConcatPositiveData))]
    // public void TestCombineTransposeConcatPositive(int[] inShape, int[] perm, int axis, int concatNum)
    // {
    //     var caseOptions = GetPassOptions();
    //     var inputList = new List<Var>();
    //     for (int i = 0; i < concatNum; i++)
    //     {
    //         inputList.Add(new Var());
    //     }
    //     var Normal = new Dictionary<Var, IValue>();
    //     var tpList = new List<Call>();
    //     foreach (Var a in inputList)
    //     {
    //         // TODO:  Rely type infer and cse
    //         // Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
    //         // tpList.Add(Tensors.Transpose(a, perm));
    //         var b = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
    //         tpList.Add(Tensors.Transpose(b, perm));
    //     }
    //
    //     var input = Enumerable.Range(0, concatNum).Select(i => tpList[i]);
    //     var rootPre = Tensors.Concat(new IR.Tuple(input), axis);
    //     var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
    //     {
    //         new CombineTransposeConcat(),
    //         // Should not open constant fold.
    //         // new FoldConstCall(),
    //     }, caseOptions);
    //
    //     Assert.NotEqual(rootPre, rootPost);
    //     Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    //     // Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    // }

    public static IEnumerable<object[]> TestCombineTransposePadPositiveData =>
        new[]
        {
            new object[] { new[] {1, 2, 3, 4}, new[] {0, 2, 3, 1}, new[,] {{ 4, 4 },
            { 3, 3 },
            { 2, 2 },
            { 1, 1 }}, PadMode.Constant, 1f},
            new object[] { new[] {1, 2, 3, 4}, new[] {0, 3, 1, 2}, new[,] {{ 1, 1 },
            { 0, 0 },
            { 1, 1 },
            { 1, 1 }}, PadMode.Symmetric, 0f},
            new object[] { new[] {5, 2, 3, 4}, new[] {3, 0, 1, 2}, new[,] {{ 2, 2 },
            { 0, 0 },
            { 1, 1 },
            { 1, 1 }}, PadMode.Reflect, 0f},
            new object[] { new[] {1, 2, 3, 4}, new[] {0, 3, 1, 2}, new[,] {{ 1, 1 },
            { 0, 0 },
            { -1, -1 },
            { 1, 1 }}, PadMode.Edge, 0f},

        };

    [Theory]
    [MemberData(nameof(TestCombineTransposePadPositiveData))]
    public void TestCombineTransposePadPositive(int[] inShape, int[] perm, int[,] paddings, PadMode padM, float padValue)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        // var a = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
        var rootPre = Pad(Tensors.Transpose(a, perm), paddings, padM, padValue);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldConstCall(),
            new CombineTransposePad(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        // Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    public static IEnumerable<object[]> TestCombineTransposeReducePositiveData =>
        new[]
        {
            new object[] {new[] {1, 3, 4}, new[] {0, 2, 1}, 1, 0, false},
            new object[] {new[] {1, 3, 4, 5}, new[] {0, 2, 3, 1}, 2, 1, true},
        };

    [Theory]
    [MemberData(nameof(TestCombineTransposeReducePositiveData))]
    public void TestCombineTransposeReducePositive(int[] inShape, int[] perm, int axis, int initValue, bool keepDims)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Tensors.Reduce(ReduceOp.Mean, Tensors.Transpose(a, perm), axis, initValue, keepDims);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeReduce(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    // TODO : CombineTransposeUnary
    public static IEnumerable<object[]> TestCombineTransposeUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] {1, 3, 4}, new[] {0, 2, 1}},
            new object[] { UnaryOp.Sqrt, new[] {1, 3, 4}, new[] {0, 2, 1}},
            new object[] { UnaryOp.Log, new[] {1, 3, 4, 5}, new[] {0, 2, 3, 1}},
            new object[] { UnaryOp.Abs, new[] {1, 3, 4, 5}, new[] {0, 2, 3, 1}},
        };

    [Theory]
    [MemberData(nameof(TestCombineTransposeUnaryPositiveData))]
    public void TestCombineTransposeUnaryPositive(UnaryOp opType, int[] inShape, int[] perm)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Transpose(a, perm));
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeUnary(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }
}