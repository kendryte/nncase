using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using torchF = TorchSharp.torch.nn.functional;
using Tuple = Nncase.IR.Tuple;


namespace Nncase.Tests.Evaluator
{
    public class EvaluatorTest
    {
        [Fact]
        public void TestUnary()
        {
            var a = (Const)7f;
            var tA = tensor(7f);
            var expr = -a;
            TypeInference.InferenceType(expr);
            Assert.Equal(
                -tA,
                expr.Eval());
        }

        [Fact]
        public void TestBinary()
        {
            var tA = tensor(1f);
            var tB = tA * 2;

            var a = (Const)1f;
            var b = (Const)2f;
            var expr = a * b + a;
            TypeInference.InferenceType(expr);
            Assert.Equal(
                tA * tB + tA,
                expr.Eval());
        }

        [Fact]
        public void TestConcat()
        {
            var a = Const.FromSpan<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 }));
            var b = Const.FromSpan<int>(new int[12], new Shape(new[] { 1, 3, 4 }));
            var inputList = new Tuple(a, b);
            var expr = Tensors.Concat(inputList, 0);
            TypeInference.InferenceType(expr);

            var tA = a.ToTorchTensor();
            var tB = b.ToTorchTensor();

            Assert.Equal(
                torch.cat(new[] { tA, tB }, 0),
                expr.Eval());
        }

        [Fact]
        public void TestSlice()
        {
            var input = Const.FromSpan<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
            var begin = Const.FromSpan<int>(new[] { 0, 0, 0, 0 }, new Shape(new[] { 4 }));
            var end = Const.FromSpan<int>(new[] { 1, 1, 1, 5 }, new Shape(new[] { 4 }));
            var axes = Const.FromSpan<int>(new[] { 0, 1, 2, 3 }, new Shape(new[] { 4 }));
            var strides = Const.FromSpan<int>(new[] { 1, 1, 1, 1 }, new Shape(new[] { 4 }));
            var result = Const.FromSpan<int>(Enumerable.Range(0, 5).ToArray(), new Shape(new[] { 1, 1, 1, 5 }));
            var tResult = result.ToTorchTensor();
            var expr = Tensors.Slice(input, begin, end, axes, strides);
            Assert.True(expr.InferenceType());
            Assert.Equal(
                tResult,
                expr.Eval()
                );
        }

        [Fact]
        public void TestPad()
        {
            var tinput = torch.randn(1, 1, 2, 3);
            var input = tinput.ToConst();
            var pads = Const.FromSpan<int>(new[] { 1, 1, 2, 2 }, new Shape(new[] { 2, 2 }));
            var value = Const.FromScalar<float>(1.0f);
            var expr = Tensors.Pad(input, pads, Nncase.PadMode.Constant, value);
            TypeInference.InferenceType(expr);
            var result = expr.Eval();
            var p = torchF.pad(tinput, new long[] {2, 3, 1, 4}, PaddingModes.Constant, 1.0f);
            Assert.Equal(p, result);
        }

        [Fact]
        public void TestPad2()
        {
            var tinput = torch.randn(1, 1, 2, 3);
            var input = tinput.ToConst();
            var pads = Const.FromSpan<int>(new[] { 1, 2, 2, 4, 5, 6 }, new Shape(new[] { 3, 2 }));
            var value = Const.FromScalar<float>(2.0f);
            var expr = Tensors.Pad(input, pads, Nncase.PadMode.Constant, value);
            TypeInference.InferenceType(expr);
            var result = expr.Eval();
            Assert.Equal(torchF.pad(tinput, new long[] { 5, 6, 2, 4, 1, 2 }, PaddingModes.Constant, 2.0f), result);
        }

        [Fact]
        public void TestStackAndCast()
        {
            var padh_before = Tensors.Cast(Const.FromSpan<float>(new[] { 1.0f }), Nncase.DataType.Int32);
            var padh_after = Tensors.Cast(Const.FromSpan<float>(new[] { 2.0f }), Nncase.DataType.Int32);
            var padw_before = Tensors.Cast(Const.FromSpan<float>(new[] { 3.0f }), Nncase.DataType.Int32);
            var padw_after = Tensors.Cast(Const.FromSpan<float>(new[] { 4.0f }), Nncase.DataType.Int32);

            var expr = Tensors.Stack(new Tuple(
              Tensors.Concat(new Tuple(padh_before, padh_after), 0),
              Tensors.Concat(new Tuple(padw_before, padw_after), 0)), 0);
            TypeInference.InferenceType(expr);
            var result = expr.Eval();
            Assert.Equal(torch.tensor(new[] { 1, 2, 3, 4 }, new long[] { 2, 2 }), result);
        }

        [Fact]
        public void TestConv2D()
        {
            var weights = torch.randn(8, 4, 3, 3);
            var inputs = torch.randn(1, 4, 5, 5);
            var bias = torch.rand(8);
            var output = torchF.conv2d(torchF.pad(inputs, new long[]{1, 0, 0, 0}), weights, bias, padding: new long[] { 1, 1 });

            var expr = NN.Conv2D(inputs.ToConst(), weights.ToConst(), bias.ToConst(),
                     stride: new[] { 1, 1 }, padding: Const.FromSpan<int>(new int[] { 1, 1, 1, 1 }, new[] { 2, 2 }),
                     dilation: new[] { 1, 1 }, Nncase.PadMode.Constant, 1);
            Assert.True(expr.InferenceType());
            Assert.Equal(output, expr.Eval());
        }

        [Fact]
        public void TestConv2D_1()
        {
            var input = torch.rand(1, 28, 28, 3).ToConst();
            var conv1 = Tensors.NCHWToNHWC(ReWrite.DummyOp.Conv2D(Tensors.NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
            Assert.True(conv1.InferenceType());
            Assert.Equal(new long[] { 1, 14, 14, 8 }, conv1.Eval().shape);
        }
    }
}