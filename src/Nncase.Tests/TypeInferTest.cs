using Xunit;
using Nncase;
using Nncase.IR;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Linq;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.Utility;

public class UnitTestTypeInfer
{
    [Fact]
    public void TestInferBinary()
    {
        Var a = new Var(new TensorType(DataType.Float32, new[] { 1, 5, 1 }));
        Const b = (Const)(new DenseTensor<float>(Enumerable.Repeat(1.0f, 15).ToArray(), new[] { 1, 5, 3 }));
        var c = a + b;
        var ctype = TypeInference.InferenceType(c);

        Assert.True(HasShape(new[] { 1, 5, 3 }).MatchLeaf(c.CheckedType));
    }

    [Fact]
    public void TestInferUnary()
    {
        Var a = new Var(AnyType.Default);
        var c = Square(a);
        Assert.Throws<TypeInferenceInterruptException>(() => TypeInference.InferenceType(c));
    }

}