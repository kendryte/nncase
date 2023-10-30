// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using ITuple = Nncase.IR.ITuple;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestBatchNormToBinary : TransformTestBase
{
    public static readonly TheoryData<int[]> BatchNormToBinaryPositiveData = new()
    {
        new[] { 56, 56 },
        new[] { 1, 64, 112, 112 },
    };

    public static readonly TheoryData<int[]> BatchNormToBinaryNegativeData = new()
    {
        new[] { 56 },
    };

    [Theory]
    [MemberData(nameof(BatchNormToBinaryPositiveData))]
    public void TestBatchNormToBinaryPositive(int[] shape)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, shape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, shape).Evaluate());
        var oc = shape[1];
        var bn = IR.F.NN.BatchNormalization(
            a,
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            0.001f,
            0.001f);
        TestMatched<BatchNormToBinary>(bn, normal);
    }

    [Theory]
    [MemberData(nameof(BatchNormToBinaryNegativeData))]
    public void TestBatchNormToBinaryNegative(int[] shape)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, shape));
        var oc = 1;
        var bn = IR.F.NN.BatchNormalization(
            a,
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { oc }).Evaluate().AsTensor(),
            0.001f,
            0.001f);
        TestNotMatch<BatchNormToBinary>(bn);
    }
}
