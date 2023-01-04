// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Autofac;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.TestFixture;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;

using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorNN : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestActivationCelu()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var alpha = 0.8F;

        var expect = OrtKI.Celu(ort_tensor, alpha);
        var expr = IR.F.NN.Celu(input, alpha);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationElu()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var alpha = 0.8F;

        var expect = OrtKI.Elu(ort_tensor, alpha);
        var expr = IR.F.NN.Elu(input, alpha);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationHardSwish()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var alpha = 1F / 6F;
        var beta = 0.5F;

        var expect = ort_tensor * OrtKI.HardSigmoid(ort_tensor, alpha, beta);
        var expr = IR.F.NN.HardSwish(input);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationLeakyRelu()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var alpha = 0.6F;

        var expect = OrtKI.LeakyRelu(ort_tensor, alpha);
        var expr = IR.F.NN.LeakyRelu(input, alpha);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationRelu()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var expect = OrtKI.Relu(ort_tensor);
        var expr = IR.F.NN.Relu(input);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationRelu6()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var expect = OrtKI.Clip(ort_tensor, 0F, 6F);
        var expr = IR.F.NN.Relu6(input);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationSelu()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var alpha = 1.2F;
        var gamma = 1.3F;
        var expect = OrtKI.Selu(ort_tensor, alpha, gamma);
        var expr = IR.F.NN.Selu(input, alpha, gamma);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationSigmoid()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var expect = OrtKI.Sigmoid(ort_tensor);
        var expr = IR.F.NN.Sigmoid(input);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationHardSigmoid()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var alpha = 1.2F;
        var gamma = 1.3F;
        var expect = OrtKI.HardSigmoid(ort_tensor, alpha, gamma);
        var expr = IR.F.NN.HardSigmoid(input, alpha, gamma);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationPRelu()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_tensor.ToTensor();
        var slope = 0.2F;
        var expect = OrtKI.PRelu(ort_tensor, slope);
        var expr = IR.F.NN.PRelu(input, slope);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestBatchToSpace()
    {
        var input = new float[] { 1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16 };
        var input_tensor = Tensor.From(input, new[] { 4, 1, 2, 2 });
        var block_shape = new long[] { 2, 2 };

        var output = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var expect = Tensor.From(output, new[] { 1, 1, 4, 4 });
        var crops = new long[] { 0, 0, 0, 0 };
        var expr = IR.F.NN.BatchToSpace(input_tensor, Tensor.From(block_shape, new[] { 2 }),
            Tensor.From(crops, new[] { 2, 2 }));
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestConv2D()
    {
        var input = OrtKI.Random(1, 4, 5, 5);
        var weight = OrtKI.Random(8, 4, 3, 3);
        var bias = OrtKI.Random(8);
        var expect = OrtKI.Conv(input, weight, bias, "NOTSET", new long[] { 1, 1 }, 1,
            new long[] { 3, 3 }, new long[] { 1, 1, 1, 1 }, new long[] { 1, 1 });

        var expr = IR.F.NN.Conv2D(input.ToTensor(), weight.ToTensor(), bias.ToTensor(),
            stride: new[] { 1, 1 }, padding: Tensor.From<int>(new int[] { 1, 1, 1, 1 }, new[] { 2, 2 }),
            dilation: new[] { 1, 1 }, Nncase.PadMode.Constant, 1);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConv2D_1()
    {
        var input = OrtKI.Random(1, 4, 5, 5);
        var weight = OrtKI.Random(8, 4, 3, 3);
        var bias = OrtKI.Random(8);
        var min = 0.5F;
        var max = 1F;
        var conv = OrtKI.Conv(input, weight, bias, "NOTSET", new long[] { 1, 1 }, 1,
            new long[] { 3, 3 }, new long[] { 1, 1, 1, 1 }, new long[] { 1, 1 });
        var expect = OrtKI.Clip(conv, min, max);

        var expr = IR.F.NN.Conv2D(input.ToTensor(), weight.ToTensor(), bias.ToTensor(),
            stride: new[] { 1, 1 }, padding: Tensor.From<int>(new int[] { 1, 1, 1, 1 }, new[] { 2, 2 }),
            dilation: new[] { 1, 1 }, Nncase.PadMode.Constant, 1, new[] { min, max });
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConv2DTranspose()
    {
        var input = OrtKI.Random(1, 1, 5, 5);
        var weight = OrtKI.Random(1, 2, 3, 3);
        var bias = OrtKI.Random(2);
        var expect = OrtKI.ConvTranspose(input, weight, bias, "NOTSET", new long[] { 1, 1 }, 1,
            kernel_shape: new long[] { 3, 3 }, output_padding: new long[] { 0, 0 }, output_shape: new long[] { 1, 2, 5, 5 },
            pads: new long[] { 1, 1, 1, 1 }, strides: new long[] { 1, 1 });

        var outShape = Tensor.From(new long[] { 1, 2, 5, 5 }, new Shape(4));
        var expr = IR.F.NN.Conv2DTranspose(input.ToTensor(), weight.ToTensor(), bias.ToTensor(), outShape,
            stride: new[] { 1, 1 }, padding: Tensor.From<long>(new long[] { 1, 1, 1, 1 }, new[] { 4 }),
            outputPadding: Tensor.From<long>(new long[] { 0, 0 }, new[] { 2 }),
            dilation: new[] { 1, 1 }, Nncase.PadMode.Constant, 1);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestHardmax()
    {
        var ort_input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input = ort_input.ToTensor();
        DoHardmax(ort_input, input, -1L);
        DoHardmax(ort_input, input, 1L);
    }

    [Fact]
    public void TestL2Normalization()
    {
        var a = new float[] { 0F, 2F, 3F, 2F, 2F, 2F };
        var b = new float[] { 0F, 0.4F, 0.6F, 0.4F, 0.4F, 0.4F };
        {
            var expect = Tensor.From(b, new[] { 6 });
            var input = Tensor.From(a, new[] { 6 });
            DoL2Normalization(expect, input);
        }

        {
            var expect = Tensor.From(b, new[] { 1, 2, 3 });
            var input = Tensor.From(a, new[] { 1, 2, 3 });
            DoL2Normalization(expect, input);
        }
    }

    [Fact]
    public void TestBatchNormalization()
    {
        var input_shape = new long[] { 1, 3, 16, 16 };
        var ort_x = OrtKI.Random(input_shape);
        var ort_scale = OrtKI.Random(new long[] { input_shape[1] });
        var ort_b = OrtKI.Random(new long[] { input_shape[1] });
        var ort_mean = OrtKI.Random(new long[] { input_shape[1] });
        var ort_var = OrtKI.Random(new long[] { input_shape[1] });
        var epsilon = 0.01F;
        var momentum = 0.5F;

        var expect = OrtKI.BatchNormalization(ort_x, ort_scale, ort_b,
            ort_mean, ort_var, epsilon, momentum);
        var expr = IR.F.NN.BatchNormalization(ort_x.ToTensor(), ort_scale.ToTensor(),
            ort_b.ToTensor(), ort_mean.ToTensor(), ort_var.ToTensor(),
            epsilon, momentum);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestInstanceNormalization()
    {
        var input_shape = new long[] { 1, 3, 16, 16 };
        var ort_x = OrtKI.Random(input_shape);
        var ort_scale = OrtKI.Random(new long[] { input_shape[1] });
        var ort_b = OrtKI.Random(new long[] { input_shape[1] });
        var epsilon = 0.01F;

        var expect = OrtKI.InstanceNormalization(ort_x, ort_scale, ort_b,
            epsilon);
        var expr = IR.F.NN.InstanceNormalization(ort_x.ToTensor(), ort_scale.ToTensor(),
            ort_b.ToTensor(), epsilon);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestLpNormalization()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        DoLpNormalization(input, 0, 1);
        DoLpNormalization(input, 0, 2);
        DoLpNormalization(input, 1, 1);
        DoLpNormalization(input, 1, 2);
    }

    [Fact]
    public void TestLRN()
    {
        var input_shape = new long[] { 1, 3, 16, 16 };
        var ort_x = OrtKI.Random(input_shape);
        var alpha = 0.001F;
        var beta = 0.5F;
        var bias = 0.8F;
        var size = 3L;

        var expect = OrtKI.LRN(ort_x, alpha, beta, bias, size);
        var expr = IR.F.NN.LRN(ort_x.ToTensor(), alpha, beta, bias, size);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestOneHotTF()
    {
        var a = new int[] { 1, 2, 0, 3 };
        var indices = Tensor.From(a, new[] { 4 });
        var depth = 5;
        var values = Tensor.From(new int[] { 0, 1 }, new Shape(new[] { 2 }));
        var axis = 0L;

        var b = new float[] { 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
        var expect = OrtKISharp.Tensor.MakeTensor(b, new long[] { 5, 4 });

        var expr = NN.OneHot(OneHotMode.Normal, indices, depth, values, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestOneHotOnnx()
    {
        var a = new float[] { 1, 2, 0, 3 };
        var indices = Tensor.From(a, new[] { 4 });
        var depth = 5F;
        var values = Tensor.From(new float[] { 0, 1 }, new Shape(new[] { 2 }));
        var axis = 1L;

        var expect = OrtKI.OneHot(indices.ToOrtTensor(), depth, values.ToOrtTensor(), axis);

        var expr = NN.OneHot(OneHotMode.ProcessNeg, indices, depth, values, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPad()
    {
        var tinput = OrtKI.Random(1, 1, 2, 3);
        var input = tinput.ToTensor();
        var pads = Tensor.From<int>(new[] { 0, 0, 0, 0, 1, 1, 2, 2 }, new Shape(new[] { 4, 2 }));
        var value = Tensor.FromScalar<float>(1.0f);
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(new long[] { 1, 1, 4, 7 }, result.Shape);
    }

    [Fact]
    public void TestPad2()
    {
        var tinput = OrtKI.Random(1, 1, 2, 3);
        var input = tinput.ToTensor();
        var pads = Tensor.From<long>(new long[] { 0, 0, 1, 2, 2, 4, 5, 6 }, new Shape(4, 2));
        var value = Tensor.FromScalar<float>(2.0f);
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(new long[] { 1, 4, 8, 14 }, result.Shape);
    }

    [Fact]
    public void TestPadConstant()
    {
        var input_shape = new long[] { 1, 3, 16, 16 };
        var ort_input = OrtKI.Random(input_shape);
        var ort_pads = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, new long[] { 8 });
        var ort_constant = OrtKISharp.Tensor.FromScalar(1F);
        var expect = OrtKI.Pad(ort_input, ort_pads, ort_constant, "constant");

        var input = ort_input.ToTensor();
        var pads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new Shape(4, 2));
        var constant = ort_constant.ToTensor();
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, constant);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPadReflect()
    {
        var input_shape = new long[] { 1, 3, 16, 16 };
        var ort_input = OrtKI.Random(input_shape);
        var ort_pads = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, new long[] { 8 });
        var ort_constant = OrtKISharp.Tensor.FromScalar(1F);
        var expect = OrtKI.Pad(ort_input, ort_pads, ort_constant, "reflect");

        var input = ort_input.ToTensor();
        var pads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new Shape(4, 2));
        var constant = ort_constant.ToTensor();
        var expr = NN.Pad(input, pads, Nncase.PadMode.Reflect, constant);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPadSymmetric()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        var b = new float[] { 1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 4, 4, 5, 6, 6 };
        var expect = OrtKISharp.Tensor.MakeTensor(b, new long[] { 1, 1, 4, 5 });

        var input = Tensor.From(a, new Shape(1, 1, 2, 3));
        var pads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new Shape(4, 2));
        var constant = 0F;
        var expr = NN.Pad(input, pads, Nncase.PadMode.Symmetric, constant);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPadEdge()
    {
        var input_shape = new long[] { 1, 3, 16, 16 };
        var ort_input = OrtKI.Random(input_shape);
        var ort_pads = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, new long[] { 8 });
        var ort_constant = OrtKISharp.Tensor.FromScalar(1F);
        var expect = OrtKI.Pad(ort_input, ort_pads, ort_constant, "edge");

        var input = ort_input.ToTensor();
        var pads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new Shape(4, 2));
        var constant = ort_constant.ToTensor();
        var expr = NN.Pad(input, pads, Nncase.PadMode.Edge, constant);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestLogSoftmax()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input_tensor = ort_tensor.ToTensor();
        DoLogSoftmax(ort_tensor, input_tensor, -1L);
        DoLogSoftmax(ort_tensor, input_tensor, 1L);
    }

    [Fact]
    public void TestReduceWindow2DMean()
    {
        var ort_input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ceilMode = false;
        var countIncludePad = false;
        var dilations = new long[] { 1, 1 };
        var filter = new long[] { 3, 3 };
        var stride = new long[] { 1, 1 };
        var onnxPads = new long[] { 1, 1, 1, 1 };
        var expect = OrtKI.AveragePool(ort_input, "NOTSET", ceilMode ? 1 : 0, countIncludePad ? 1 : 0,
            filter, onnxPads, stride);

        var input = ort_input.ToTensor();
        var expr = IR.F.NN.ReduceWindow2D(ReduceOp.Mean, input, 0.0f, filter, stride, new[,]
            {
                { 1, 1 },
                { 1, 1 },
            }, dilations, false, false);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestReduceWindow2DMax()
    {
        var ort_input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ceilMode = false;
        var dilations = new long[] { 1, 1 };
        var filter = new long[] { 3, 3 };
        var stride = new long[] { 1, 1 };
        var onnxPads = new long[] { 1, 1, 1, 1 };
        var storage_order = 0L;
        var expect = OrtKI.MaxPool(ort_input, "NOTSET", ceilMode ? 1 : 0, dilations,
            filter, onnxPads, storage_order, stride)[0];

        var input = ort_input.ToTensor();
        var expr = IR.F.NN.ReduceWindow2D(ReduceOp.Max, input, 0.0f, filter, stride, new[,]
            {
                { 1, 1 },
                { 1, 1 },
            }, dilations, false, false);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSoftmax()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input_tensor = ort_tensor.ToTensor();
        DoSoftmax(ort_tensor, input_tensor, -1);
        DoSoftmax(ort_tensor, input_tensor, 1);
    }

    [Fact]
    public void TestSoftplus()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input_tensor = ort_tensor.ToTensor();
        var expect = OrtKI.Softplus(ort_tensor);
        var expr = IR.F.NN.Softplus(input_tensor);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSoftsign()
    {
        var ort_tensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var input_tensor = ort_tensor.ToTensor();
        var expect = OrtKI.Softsign(ort_tensor);
        var expr = IR.F.NN.Softsign(input_tensor);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSpaceToBatch()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var input_tensor = Tensor.From(input, new[] { 1, 4, 4, 1 });
        var block_shape = new long[] { 2, 2 };

        var output = new float[] { 1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16 };
        var expect = Tensor.From(output, new[] { 4, 2, 2, 1 });
        var crops = new long[] { 0, 0, 0, 0 };
        var expr = IR.F.NN.SpaceToBatch(input_tensor, Tensor.From(block_shape, new[] { 2 }),
            Tensor.From(crops, new[] { 2, 2 }));
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    private void DoHardmax(OrtKISharp.Tensor ort_tensor, Tensor input_tensor, long axis)
    {
        var expect = OrtKI.Hardmax(ort_tensor, axis);
        var expr = IR.F.NN.Hardmax(input_tensor, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void DoLogSoftmax(OrtKISharp.Tensor ort_tensor, Tensor input_tensor, long axis)
    {
        var expect = OrtKI.LogSoftmax(ort_tensor, axis);
        var expr = IR.F.NN.LogSoftmax(input_tensor, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void DoSoftmax(OrtKISharp.Tensor ort_tensor, Tensor input_tensor, int axis)
    {
        var expect = OrtKI.Softmax(ort_tensor, axis);
        var expr = IR.F.NN.Softmax(input_tensor, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void DoL2Normalization(Tensor expect, Tensor input)
    {
        var expr = IR.F.NN.L2Normalization(input);
        CompilerServices.InferenceType(expr);

        // fix precision issue on Macos
        var cos = Comparator.CosSimilarity(expect, expr.Evaluate().AsTensor());
        Assert.True(cos > 0.999F);
    }

    private void DoLpNormalization(OrtKISharp.Tensor input, long axis, long p)
    {
        var expect = OrtKI.LpNormalization(input, axis, p);
        var expr = IR.F.NN.LpNormalization(input.ToTensor(), axis, p);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }
}
