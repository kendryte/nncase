using System.IO;
using System.Linq;
using Autofac;
using Autofac.Extras.CommonServiceLocator;
using CommonServiceLocator;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using TorchSharp;
using Nncase.IR;
using OrtKISharp;
using Xunit;
using static TorchSharp.torch;
using torchF = TorchSharp.torch.nn.functional;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest
{
    public class UnitTestEvaluator : IHostFixtrue
    {
        public UnitTestEvaluator(IHost host) : base(host)
        {
            OrtKI.LoadDLL();
        }

        [Fact]
        public void TestOrtKI()
        {
            var a = Const.FromTensor(Tensor.FromSpan<int>(new[] {1, 2, 3}));
            var b = Const.FromTensor(Tensor.FromSpan<int>(new[] {1, 2, 3}));
            // var b = (Const) 2;
            a.InferenceType();
            b.InferenceType();
            var na = a.Value.ToOrtTensor();
            var nb = b.Value.ToOrtTensor();
            Assert.Equal(new[] {1, 2, 3}, na.ToDense<int>().ToArray());
            var v = na.ToType(OrtDataType.Float16).ToValue();
            var f = na.ToType(OrtDataType.Float16).ToType(OrtDataType.Float);
            
            var c = na + nb;
            Assert.Equal(new[] {2, 4, 6}, c.ToTensor().ToArray<int>());
        }
        
        [Fact]
        public void TestUnary()
        {
            var a = (Const)7f;
            var tA = tensor(7f);
            var expr = -a;
            CompilerServices.InferenceType(expr);
            Assert.Equal(
                -tA,
                expr.Evaluate().AsTensor().ToTorchTensor());
        }

        [Fact]
        public void TestBinary()
        {
            var tA = tensor(1f);
            var tB = tA * 2;

            var a = (Const)1f;
            var b = (Const)2f;
            var expr = a * b + a;
            CompilerServices.InferenceType(expr);
            Assert.Equal(
                tA * tB + tA,
                expr.Evaluate().AsTensor().ToTorchTensor());
        }

        [Fact]
        public void TestConcat()
        {
            var a = Const.FromTensor(Tensor.FromSpan<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 })));
            var b = Const.FromTensor(Tensor.FromSpan<int>(new int[12], new Shape(new[] { 1, 3, 4 })));
            var inputList = new Tuple(a, b);
            var expr = Tensors.Concat(inputList, 0);
            CompilerServices.InferenceType(expr);

            var tA = a.Value.ToTorchTensor();
            var tB = b.Value.ToTorchTensor();

            Assert.Equal(
                torch.cat(new[] { tA, tB }, 0),
                expr.Evaluate().AsTensor().ToTorchTensor());
        }

        [Fact]
        public void TestSlice()
        {
            var input = Tensor.FromSpan<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
            var begin = Tensor.FromSpan<int>(new[] { 0, 0, 0, 0 }, new Shape(new[] { 4 }));
            var end = Tensor.FromSpan<int>(new[] { 1, 1, 1, 5 }, new Shape(new[] { 4 }));
            var axes = Tensor.FromSpan<int>(new[] { 0, 1, 2, 3 }, new Shape(new[] { 4 }));
            var strides = Tensor.FromSpan<int>(new[] { 1, 1, 1, 1 }, new Shape(new[] { 4 }));
            var result = Const.FromTensor(Tensor.FromSpan<int>(Enumerable.Range(0, 5).ToArray(), new Shape(new[] { 1, 1, 1, 5 })));
            var tResult = result.Value.ToTorchTensor();
            var expr = Tensors.Slice(input, begin, end, axes, strides);
            Assert.True(expr.InferenceType());
            Assert.Equal(
                tResult,
                expr.Evaluate().AsTensor().ToTorchTensor()
                );
        }

        [Fact]
        public void TestPad()
        {
            var tinput = torch.randn(1, 1, 2, 3);
            var input = tinput.ToTensor();
            var pads = Tensor.FromSpan<int>(new[] { 0, 0, 0, 0, 1, 1, 2, 2 }, new Shape(new[] { 4, 2 }));
            var value = Tensor.FromScalar<float>(1.0f);
            var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
            CompilerServices.InferenceType(expr);
            var result = expr.Evaluate().AsTensor().ToTorchTensor();
            Assert.Equal(new long[] { 1, 1, 4, 7 }, result.shape);
        }

        [Fact]
        public void TestPad2()
        {
            var tinput = torch.randn(1, 1, 2, 3);
            var input = tinput.ToTensor();
            var pads = Tensor.FromSpan<int>(new[] { 0, 0, 1, 2, 2, 4, 5, 6 }, new Shape(new[] { 4, 2 }));
            var value = Tensor.FromScalar<float>(2.0f);
            var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
            CompilerServices.InferenceType(expr);
            var result = expr.Evaluate().AsTensor().ToTorchTensor();
            Assert.Equal(new long[] { 1, 4, 8, 14 }, result.shape);
        }

        [Fact]
        public void TestStackAndCast()
        {
            var padh_before = Tensors.Cast(Tensor.FromSpan<float>(new[] { 1.0f }), Nncase.DataTypes.Int32);
            var padh_after = Tensors.Cast(Tensor.FromSpan<float>(new[] { 2.0f }), Nncase.DataTypes.Int32);
            var padw_before = Tensors.Cast(Tensor.FromSpan<float>(new[] { 3.0f }), Nncase.DataTypes.Int32);
            var padw_after = Tensors.Cast(Tensor.FromSpan<float>(new[] { 4.0f }), Nncase.DataTypes.Int32);

            var expr = Tensors.Stack(new Tuple(
              Tensors.Concat(new Tuple(padh_before, padh_after), 0),
              Tensors.Concat(new Tuple(padw_before, padw_after), 0)), 0);
            CompilerServices.InferenceType(expr);
            var result = expr.Evaluate().AsTensor().ToTorchTensor();
            Assert.Equal(torch.tensor(new[] { 1, 2, 3, 4 }, new long[] { 2, 2 }), result);
        }

        [Fact]
        public void TestConv2D()
        {
            var weights = torch.randn(8, 4, 3, 3);
            var inputs = torch.randn(1, 4, 5, 5);
            var bias = torch.rand(8);
            var output = torchF.conv2d(inputs, weights, bias, padding: new long[] { 1, 1 });

            var expr = NN.Conv2D(inputs.ToTensor(), weights.ToTensor(), bias.ToTensor(),
                stride: new[] { 1, 1 }, padding: Tensor.FromSpan<int>(new int[] { 1, 1, 1, 1 }, new[] { 2, 2 }),
                dilation: new[] { 1, 1 }, Nncase.PadMode.Constant, 1);
            Assert.True(expr.InferenceType());
            Assert.Equal(output, expr.Evaluate().AsTensor().ToTorchTensor());
        }

        [Fact]
        public void TestConv2D_1()
        {
            // var input = torch.rand(1, 28, 28, 3).ToTensor();
            // var conv1 = Tensors.NCHWToNHWC(ReWriteTest.DummyOp.Conv2D(Tensors.NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
            // Assert.True(conv1.InferenceType());
            // Assert.Equal(new long[] { 1, 14, 14, 8 }, conv1.Evaluate().AsTensor().ToTorchTensor().shape);
        }

        [Fact]
        public void TestProd()
        {
            var input = Tensor.FromSpan<int>(new[] { 1, 2, 3, 4 });
            var prod = Tensors.Prod(input);
            prod.InferenceType();
            Assert.Equal(1 * 2 * 3 * 4, prod.Evaluate().AsTensor().ToScalar<int>());
        }

        [Fact]
        public void TestSize()
        {
            var input = torch.rand(1, 3, 224, 224).ToTensor();
            var size = Tensors.SizeOf(input);
            size.InferenceType();
            Assert.Equal(1 * 3 * 224 * 224, size.Evaluate().AsTensor().ToScalar<int>());
        }
    }
}