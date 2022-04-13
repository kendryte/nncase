using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Nncase.Transform.Rules.Neutral;
using Tensorflow.Operations.Initializers;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestIntegralPromotion
{
    private readonly RunPassOptions passOptions;

    public UnitTestIntegralPromotion()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
    }

    public static IEnumerable<object[]> TestIntegralPromotionPositiveData =>
        new[]
        {
            new object[] {DataTypes.Int32, DataTypes.Int64},
            new object[] {DataTypes.Int64, DataTypes.Int32},
        };

    [Theory]
    [MemberData(nameof(TestIntegralPromotionPositiveData))]
    public void TestIntegralPromotionPositive(DataType aType, DataType bType)
    {
        var expr = Tensors.Cast(1, aType) + Tensors.Cast(2, bType);
        expr.InferenceType();
        var f = new Function(expr);
        var result = CompilerServices.InferenceType(f);
        Assert.False(result);
        CompilerServices.DumpIR(f, "before", Path.Combine(passOptions.PassDumpDir, "TypePromotion"));
        var post = new ShapeInferPass("TypePromotion").Run(f, passOptions);
        Assert.True(CompilerServices.InferenceType(post));
        Assert.Equal(Value.FromTensor(3L), ((Function) post).Body.Evaluate());
    }
}