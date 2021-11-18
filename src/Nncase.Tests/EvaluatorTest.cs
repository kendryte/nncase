using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;
using Tuple = Nncase.IR.Tuple;


public class EvaluatorTest
{
    [Fact]
    public void TestUnary()
    {
        var a = (Const) 7f;
        var tA = tensor(7f);

        Assert.Equal(
            ~tA,
            Evaluator.Eval(~a));
    }
    
    [Fact]
    public void TestBinary()
    {
        var tA = tensor(1f);
        var tB = tA * 2;
        
        var a = (Const) 1f;
        var b = (Const) 2f;
        
        Assert.Equal(
            tA * tB + tA, 
            Evaluator.Eval(a * b + a));
    }

    [Fact]
    public void TestConcat()
    {
        var a = Const.FromSpan<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] {1, 3, 4}));
        var b = Const.FromSpan<int>(new int[12], new Shape(new[] {1, 3, 4}));

        var tA = Util.ToTorchTensor(a);
        var tB = Util.ToTorchTensor(b);
        var inputList = new Tuple(a, b);
        Assert.Equal(
            torch.cat(new[] {tA, tB}, 0),
            Evaluator.Eval(Tensors.Concat(inputList, 0)));
    }

    [Fact]
    public void TestSlice()
    {
        var input = Const.FromSpan<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] {2, 3, 4, 5}));
        var begin = Const.FromSpan<int>(new[] {0, 0, 0, 0}, new Shape(new[] {4}));
        var end = Const.FromSpan<int>(new[] {1, 1, 1, 5}, new Shape(new[] {4}));
        var axes = Const.FromSpan<int>(new[] {0, 1, 2, 3}, new Shape(new[] {4}));
        var strides = Const.FromSpan<int>(new[] {1, 1, 1, 1}, new Shape(new[] {4}));
        TypeInference.InferenceType(input);
        TypeInference.InferenceType(begin);
        TypeInference.InferenceType(end);
        TypeInference.InferenceType(axes);
        TypeInference.InferenceType(strides);
        
        var result = Const.FromSpan<int>(Enumerable.Range(0, 5).ToArray(), new Shape(new[] {1, 1, 1, 5}));        TypeInference.InferenceType(strides);
        TypeInference.InferenceType(result);
        var tResult = Util.ToTorchTensor(result);
        Assert.Equal(
            tResult,
            Evaluator.Eval(Tensors.Slice(input, begin, end, axes, strides)
            ));
    }
}