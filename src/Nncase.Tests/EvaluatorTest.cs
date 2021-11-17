using System.Linq;
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
}