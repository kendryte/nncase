using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Passes;
using OrtKISharp;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.ReWriteTest;

public static class DummyOp
{
    public static Call Conv2D(Expr input, int in_channels, int out_channels, int kernel = 3, int stride = 1)
    {
        var weights = OrtKI.Random(out_channels, in_channels, kernel, kernel).ToTensor();
        var bias = OrtKI.Random(out_channels).ToTensor();
        return IR.F.NN.Conv2D(input, weights, bias, new[] { stride, stride }, Tensor.From<int>(new[] { 1, 1, 1, 1 }, new[] { 2, 2 }), new[] { 1, 1 }, PadMode.Constant, 1);
    }
}

public class RewriteFixtrue : TestFixture.UnitTestFixtrue
{
    public async Task<Expr> RunShapeInferPass(string name, RunPassOptions caseOptions, Expr expr, params Var[] parameters)
    {
        expr.InferenceType();
        var f = new Function(expr, parameters);
        var result = CompilerServices.InferenceType(f);
        CompilerServices.DumpIR(f, "before", Path.Combine(caseOptions.PassDumpDir, $"ShapeInfer_{name}"));
        return ((Function)await new ShapeInferPass($"ShapeInfer_{name}").RunAsync(f, caseOptions)).Body;
    }

    public Expr ApplyFoldConstCallRewrite(Expr expr, RunPassOptions caseOptions) =>
        CompilerServices.Rewrite(expr, new[] { new Transform.Rules.Neutral.FoldConstCall() }, caseOptions);
}

public interface IRewriteCase
{
    /// <summary>
    /// Get Name
    /// </summary>
    string Name => this.GetType().Name;

    /// <summary>
    /// Get Pre Expr
    /// </summary>
    Function PreExpr { get; }

    /// <summary>
    /// get rules
    /// </summary>
    IEnumerable<IRewriteRule> Rules { get; }

    /// <summary>
    /// the eval inputs dict
    /// </summary>
    Dictionary<Var, IValue> FeedDict { get; }
}

// public sealed class TransposeConstBinaryCase : IRewriteCase
// {
//     public Function PreExpr
//     {
//         get
//         {
//             var c = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 2, 3, 4 }).Evaluate().AsTensor();
//             var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
//             var b = NHWCToNCHW(input) + c;
//             return new Function(b, new Var[] { });
//         }
//     }

//     public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//     };
// }

public sealed class FoldReshapeCase : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var b = Reshape(input, (Const)new[] { 1, 1, 1, 6 });
            var c = Reshape(input, (Const)new[] { 1, 1, 3, 2 });
            return new Function(c, new Var[] { });
        }
    }

    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldTwoReshapes(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}

public sealed class FoldNopReshapeCase : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var b = Reshape(input, (Const)new[] { 1, 3, 1, 2 });
            return new Function(b, new Var[] { });
        }
    }

    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldNopReshape(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}

public sealed class FoldNopClampCase : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var b = Clamp(input, float.MinValue, float.MaxValue);
            return new Function(b, new Var[] { });
        }
    }

    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldNopClamp(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}

public class FoldTransposeCase : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var b = NHWCToNCHW(input); // [1,2,3,1]
            var d = NCHWToNHWC(b) * IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 4, 2 }).Evaluate().AsTensor();
            var e = d + 100.0f;
            return new Function(e, new Var[] { });
        }
    }

    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldTwoTransposes(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}


/// <summary>
/// transpose + pad + transpose => pad.
/// </summary>
public class FoldTransposePadCase : IRewriteCase
{
    Var _input;

    public FoldTransposePadCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 1, 2 }));
    }
    public Function PreExpr
    {
        get
        {
            var v0 = NHWCToNCHW(_input); // [1,2,3,1]
            var v1 = IR.F.NN.Pad(v0, new[,] { { 0, 0 }, { 0, 0 }, { 2, 2 }, { 1, 1 } }, PadMode.Constant, 1.0f); // [1,2,7,3]
            var v2 = NCHWToNHWC(v1); // [1,7,3,2]
            var v3 = v2 * IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 3, 7, 3, 2 }).Evaluate().AsTensor(); // [3,7,3,2]
            return new Function(v3, new Var[] { _input });
        }
    }

    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldTwoTransposes(),
      new Transform.Rules.Neutral.CombineTransposePad(),
      new Transform.Rules.Neutral.FoldConstCall(),
    };

    public Dictionary<Var, IValue> FeedDict => new(ReferenceEqualityComparer.Instance) {
      { _input, IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] {1, 3, 1, 2 }).Evaluate() }
    };

}


public class FoldNopTransposeCase1 : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var b = Transpose(input, new[] { 0, 1, 2, 3 }); // [1,2,3,1]
            var d = NCHWToNHWC(b) * IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 1, 2, 1 }).Evaluate().AsTensor();
            var e = d + 100.0f;
            return new Function(e, new Var[] { });
        }
    }
    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldTwoTransposes(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}

public class FoldNopTransposeCase2 : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var b = NHWCToNCHW(input);
            var rhs = b + IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 2, 3, 4 }).Evaluate().AsTensor();
            var lhs = NCHWToNHWC(b) - IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            var e = lhs + NCHWToNHWC(rhs);
            return new Function(e, new Var[] { });
        }
    }
    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldTwoTransposes(),
      new Transform.Rules.Neutral.FoldNopTranspose(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}

public class FoldNopTransposeCase3 : FoldNopTransposeCase2
{
    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.FoldTwoTransposes(),
      new Transform.Rules.Neutral.FoldNopTranspose(),
      new Transform.Rules.Neutral.CombineTransposeBinary(),
    };
}

public class ClassicDemo : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var x = (Const)1234;
            return new Function((x * 2) / 2, new Var[] { });
        }
    }
    public IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
      new Transform.Rules.Neutral.Xmul1(),
      new Transform.Rules.Neutral.ReassociateDiv(),
      new Transform.Rules.Neutral.ReassociateMul(),
      // new Transform.Rules.Neutral.Reassociate(),
      new Transform.Rules.Neutral.XDivX(),
    };

    public Dictionary<Var, IValue> FeedDict => new();

}


public sealed class TransposeDemoCase : FoldNopTransposeCase3
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 28, 28, 3 }).Evaluate().AsTensor();
            var conv1 = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
            var lhs = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(conv1), 8, out_channels: 8, 3, 1));
            var rhs = conv1 + IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { new long[] { 1, 14, 14, 8 } }).Evaluate().AsTensor();
            return new Function(lhs + rhs, new Var[] { });
        }
    }
}



/// <summary>
/// this case from mobilenet v1
/// </summary>
public sealed class MobileNetV1TransposeCase : IRewriteCase
{
    Var _input;
    public MobileNetV1TransposeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1,32,112,112 }));
    }
    public Function PreExpr
    {
        get
        {
            var v_5 = Transpose(_input, (new[] { 0, 2, 3, 1 })); // f32[1,112,112,32]
            var v_6 = Transpose(v_5, (new[] { 0, 3, 1, 2 })); // f32[1,32,112,112]
            var v_7 = Conv2D(v_6,
              (Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 32, 1, 1 }).Evaluate().AsTensor()),
              (Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor()), (new[] { 1, 1 }), (new[,] { { 0, 0 }, { 0, 0 } }), (new[] { 1, 1 }), PadMode.Constant, (1), (new[] { 0.0f, 6.0f })); // f32[1,64,112,112]
            var v_8 = Transpose(v_7, new[] { 0, 2, 3, 1 }); // f32[1,112,112,64]
            var v_9 = Pad(v_8, (new[,] { { 0, 0 }, { 0, 1 }, { 0, 1 }, { 0, 0 } }), PadMode.Constant, (0.0f)); // f32[1,113,113,64]
            var v_10 = Transpose(v_9, (new[] { 0, 3, 1, 2 })); // f32[1,64,113,113]
            var v_11 = Conv2D(v_10,
              (Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 1, 3, 3 }).Evaluate().AsTensor()),
              (Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor()),
              (new[] { 2, 2 }), (new[,] { { 0, 0 }, { 0, 0 } }), (new[] { 1, 1 }), PadMode.Constant, (64),
              (new[] { 0.0f, 6.0f })); // f32[1,64,56,56]
            return new Function(v_11, new Var[] { _input });
        }
    }

    public IEnumerable<IRewriteRule> Rules { get; } = new IRewriteRule[] {
        new Transform.Rules.Neutral.FoldConstCall(),
        new Transform.Rules.Neutral.FoldNopTranspose(),
        new Transform.Rules.Neutral.FoldTwoTransposes(),
        new Transform.Rules.Neutral.CombineTransposePad(),
    };

    public Dictionary<Var, IValue> FeedDict => new() {
      {_input, Normal(DataTypes.Float32, 0, 1, 1, new[] { 1,32,112,112 }).Evaluate() }
    };
}
