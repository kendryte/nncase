// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Nncase.IR;
using Nncase.Transform;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestCombineBinary : TestFixture.UnitTestFixtrue
{
    public static IEnumerable<object[]> CombineClampBinaryPositiveData
    {
        get
        {
            var inputShapes = new object[]{
              new []{1,32,24,24},
            };
            var constShapes = new object[]{
              new []{32},
              new []{24,24,32}
            };
            var clampShapes = new object[]{
              Array.Empty<int>(),
              new []{32},
              new []{24,24,32}
            };
            var mins = new object[]{
              0.0f,float.NegativeInfinity
            };
            var maxs = new object[]{
              1.0f,float.PositiveInfinity
            };
            return LinqExtensions.CartesianProduct(new[] { inputShapes, constShapes, clampShapes, mins, maxs }).Select(
              p => p.ToArray());
        }
    }

    private (Var, Expr) GetCombineClampBinaryCase(BinaryOp op, int[] inputShape, int[] constShape, int[] clampShape, float min, float max)
    {
        Expr rootPre;
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        {
            var v0 = IR.F.Tensors.NCHWToNHWC(input);
            var v1 = IR.F.Math.Binary(op, v0, Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, constShape).Evaluate()));
            var v2 = IR.F.Math.Clamp(v1, Tensor.FromScalar<float>(min, clampShape), Tensor.FromScalar<float>(max, clampShape));
            rootPre = v2;
        }
        return (input, rootPre);
    }

    private (Var, Var, Expr) GetCombineClampBinaryNegativeCase(BinaryOp op, int[] inputShape, int[] constShape, int[] clampShape, float min, float max)
    {
        Expr rootPre;
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        var constInput = new Var("constInput", new TensorType(DataTypes.Float32, constShape));
        {
            var v0 = IR.F.Tensors.NCHWToNHWC(input);
            var v1 = IR.F.Math.Binary(op, v0, constInput);
            var v2 = IR.F.Math.Clamp(v1, Tensor.FromScalar<float>(min, clampShape), Tensor.FromScalar<float>(max, clampShape));
            rootPre = v2;
        }
        return (input, constInput, rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampAddPositive(int[] inputShape, int[] constShape, int[] clampShape, float min, float max)
    {
        var caseOptions = GetPassOptions();
        var (input, rootPre) = GetCombineClampBinaryCase(BinaryOp.Add, inputShape, constShape, clampShape, min, max);

        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance){
          {input,IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate()}
        };

        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new Transform.Rules.Neutral.CombineClampAdd(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(TestFixture.Comparator.Compare(CompilerServices.Evaluate(rootPre, feedDict), CompilerServices.Evaluate(rootPost, feedDict)));
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampMulPositive(int[] inputShape, int[] constShape, int[] clampShape, float min, float max)
    {
        var caseOptions = GetPassOptions();
        var (input, rootPre) = GetCombineClampBinaryCase(BinaryOp.Mul, inputShape, constShape, clampShape, min, max);

        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance){
          {input,IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate()}
        };

        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new Transform.Rules.Neutral.CombineClampMul(),
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(TestFixture.Comparator.Compare(CompilerServices.Evaluate(rootPre, feedDict), CompilerServices.Evaluate(rootPost, feedDict)));
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampAddNegative(int[] inputShape, int[] constShape, int[] clampShape, float min, float max)
    {
        var caseOptions = GetPassOptions();
        var (input, constInput, rootPre) = GetCombineClampBinaryNegativeCase(BinaryOp.Add, inputShape, constShape, clampShape, min, max);

        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance){
          {input,IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate()},
          {constInput,IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, constShape).Evaluate()}
        };

        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new Transform.Rules.Neutral.CombineClampAdd(),
        }, caseOptions);

        Assert.Equal(rootPre, rootPost);
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampMulNegative(int[] inputShape, int[] constShape, int[] clampShape, float min, float max)
    {
        var caseOptions = GetPassOptions();
        var (input, constInput, rootPre) = GetCombineClampBinaryNegativeCase(BinaryOp.Mul, inputShape, constShape, clampShape, min, max);

        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance){
          {input,IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate()},
          {constInput,IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, constShape).Evaluate()}
        };

        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new Transform.Rules.Neutral.CombineClampMul(),
        }, caseOptions);

        Assert.Equal(rootPre, rootPost);
    }

}
