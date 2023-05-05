// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Random;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluator : TestClassBase
{
    [Fact]
    public void TestEvalFuncCall()
    {
        var cfunc = (int x) => ((x * 10) - 200) / 5;

        var x = new Var("x", TensorType.Scalar(DataTypes.Int32));
        var func = new Function("main", ((x * 10) - 200) / 5, new[] { x });

        Assert.Equal(cfunc(10), new Call(func, 10).Evaluate().AsTensor().ToScalar<int>());

        Assert.Equal(cfunc(10) + cfunc(12), (new Call(func, 10) + new Call(func, 12)).Evaluate().AsTensor().ToScalar<int>());
    }

    [Fact]
    public void TestOrtKI()
    {
        var a = Const.FromTensor(Tensor.From<int>(new[] { 1, 2, 3 }));
        var b = Const.FromTensor(Tensor.From<int>(new[] { 1, 2, 3 }));

        // var b = (Const) 2;
        a.InferenceType();
        b.InferenceType();
        var na = a.Value.ToOrtTensor();
        var nb = b.Value.ToOrtTensor();
        Assert.Equal(new[] { 1, 2, 3 }, na.ToArray<int>());
        _ = na.Cast(OrtDataType.Float16).ToValue();
        _ = na.Cast(OrtDataType.Float16).Cast(OrtDataType.Float);

        var c = na + nb;
        Assert.Equal(new[] { 2, 4, 6 }, c.ToTensor().ToArray<int>());
    }

    [Fact]
    public void TestStackAndCast()
    {
        var padh_before = Tensors.Cast(Tensor.From<float>(new[] { 1.0f }), Nncase.DataTypes.Int32);
        var padh_after = Tensors.Cast(Tensor.From<float>(new[] { 2.0f }), Nncase.DataTypes.Int32);
        var padw_before = Tensors.Cast(Tensor.From<float>(new[] { 3.0f }), Nncase.DataTypes.Int32);
        var padw_after = Tensors.Cast(Tensor.From<float>(new[] { 4.0f }), Nncase.DataTypes.Int32);

        var expr = Tensors.Stack(
            new Tuple(
                Tensors.Concat(new Tuple(padh_before, padh_after), 0),
                Tensors.Concat(new Tuple(padw_before, padw_after), 0)),
            0);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(OrtKISharp.Tensor.MakeTensor(new[] { 1, 2, 3, 4 }, new long[] { 2, 2 }), result);
    }

    [Fact]
    public void TestTFResizeImage()
    {
        var input = OrtKI.Random(1, 3, 224, 224).ToTensor();
        var image = Imaging.ResizeImage(ImageResizeMode.Bilinear, input, Array.Empty<int>(), new[] { 1, 3, 112, 112 }, isTFResize: true);
        image.InferenceType();
        Assert.Equal(new[] { 1, 3, 112, 112 }, image.Evaluate().AsTensor().Dimensions.ToArray());
    }

    [Fact]
    public void TestOnnxResizeImage()
    {
        var input = OrtKI.Random(1, 3, 224, 224).ToTensor();
        var image = Imaging.ResizeImage(ImageResizeMode.Bilinear, input, Array.Empty<float>(), new[] { 1, 3, 112, 112 }, isTFResize: false);
        image.InferenceType();
        Assert.Equal(new[] { 1, 3, 112, 112 }, image.Evaluate().AsTensor().Dimensions.ToArray());
    }

    [Fact]
    public void TestLoadStore()
    {
        var loop_i = new Var(TensorType.Scalar(DataTypes.Int32));
        var load = T.Load(T.Handle("hd", DataTypes.Float32), loop_i);
        CompilerServices.InferenceType(load);

        var store = T.Store((Var)load[TIR.Load.Handle], load[TIR.Load.Index], loop_i);
        CompilerServices.InferenceType(store);
    }

    [Fact]
    public void TestNop()
    {
        var nop = T.Nop();
        CompilerServices.InferenceType(nop);
    }

    [Fact]
    public void TestRamp()
    {
        var ramp = T.Ramp(1, 2, 0);
        CompilerServices.InferenceType(ramp);
    }

    [Fact]
    public void TestEvaluatorUtil()
    {
        var pad = OrtKI.Random(1);
        Assert.Throws<InvalidOperationException>(() => EvaluatorUtil.ToOnnxPadFormat(pad));

        var pads = OrtKI.Random(2, 2);
        var expect = OrtKI.Transpose(pads.Cast(OrtDataType.Int64), new long[] { 1, 0 }).ToArray<long>();
        Assert.Equal(expect, EvaluatorUtil.ToOnnxPadFormat(pads));
    }
}
