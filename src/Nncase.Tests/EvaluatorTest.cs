using System;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using TorchSharp;
using Xunit;
using Nncase.IR.NN;
using static TorchSharp.TensorExtensionMethods;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.TensorExtensionMethods;
using Module = Nncase.IR.Module;


public class EvaluatorTest
{
    [Fact]
    public void TestBinary()
    {
        var a = (Const) 1f;
        var b = (Const) 2f;

        var tA = tensor(1f);
        var tB = tA * 2;

        Assert.Equal(
            tA + tB, 
            Evaluator.Eval(a + b));
    }
}