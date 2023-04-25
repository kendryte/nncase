// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;

using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorRandom : TestClassBase
{
    [Fact]
    public void TestNormal()
    {
        var mean = 0.5F;
        var scale = 1F;
        var seed = 1F;
        var shape = new long[] { 1, 3, 16, 16 };

        var expect = OrtKI.RandomNormal(1, mean, scale, seed, shape);
        var expr = IR.F.Random.Normal(DataTypes.Float32, mean, scale, seed, shape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        CompilerServices.InferenceType(IR.F.Random.Normal(DataType.FromTypeCode(Runtime.TypeCode.Float32)));
    }

    [Fact]
    public void TestNormalLike()
    {
        var mean = 0.5F;
        var scale = 1F;
        var seed = 1F;

        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKISharp.Tensor.Empty(shape);
        var expect = OrtKI.RandomNormalLike(input, 1, mean, scale, seed);

        var expr = IR.F.Random.NormalLike(DataTypes.Float32, input.ToTensor(), mean, scale, seed);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestUniform()
    {
        var high = 1F;
        var low = 0F;
        var seed = 1F;
        var shape = new long[] { 1, 3, 16, 16 };

        var expect = OrtKI.RandomUniform(1, high, low, seed, shape);
        var expr = IR.F.Random.Uniform(DataTypes.Float32, high, low, seed, shape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestUniformLike()
    {
        var high = 1F;
        var low = 0F;
        var seed = 1F;

        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKISharp.Tensor.Empty(shape);
        var expect = OrtKI.RandomUniformLike(input, 1, high, low, seed);

        var expr = IR.F.Random.UniformLike(DataTypes.Float32, input.ToTensor(), high, low, seed);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }
}
