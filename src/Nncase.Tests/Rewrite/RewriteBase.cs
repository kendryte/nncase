// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.ReWriteTest;

public interface IRewriteCase
{
    /// <summary>
    /// Gets get Name.
    /// </summary>
    string Name => GetType().Name;

    /// <summary>
    /// Gets get Pre Expr.
    /// </summary>
    Function PreExpr { get; }

    /// <summary>
    /// Gets get rules.
    /// </summary>
    IEnumerable<Type> Rules { get; }

    /// <summary>
    /// Gets the eval inputs dict.
    /// </summary>
    Dictionary<Var, IValue> FeedDict { get; }

    /// <summary>
    /// check rewrite post callback.
    /// </summary>
    bool ChecksPostCallBack(Function post) => true;
}

public static class DummyOp
{
    public static Call Conv2D(Expr input, int in_channels, int out_channels, int kernel = 3, int stride = 1)
    {
        var weights = OrtKI.Random(out_channels, in_channels, kernel, kernel).ToTensor();
        var bias = OrtKI.Random(out_channels).ToTensor();
        return IR.F.NN.Conv2D(input, weights, bias, new[] { stride, stride }, Tensor.From<int>(new[] { 1, 1, 1, 1 }, new[] { 2, 2 }), new[] { 1, 1 }, PadMode.Constant, 1);
    }
}

public class RewriteFixtrue : TestClassBase
{
    public async Task<Expr> RunShapeInferPass(string name, Expr expr, params Var[] parameters)
    {
        var f = new Function(expr, parameters);
        var result = ((Function)await new ShapeInferPass { Name = $"ShapeInfer_{name}" }.RunAsync(f, new())).Body;
        Assert.True(CompilerServices.InferenceType(CompilerServices.InferenceType(f)));
        return result;
    }

    public Expr ApplyFoldConstCallRewrite(Expr expr) =>
        CompilerServices.Rewrite(expr, new[] { new Passes.Rules.Neutral.FoldConstCall() }, new());
}

public sealed class FoldReshapeCase : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate().AsTensor();
            _ = Reshape(input, (Const)new[] { 1, 1, 1, 6 });
            var c = Reshape(input, (Const)new[] { 1, 1, 3, 2 });
            return new Function(c, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldTwoReshapes),
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
            return new Function(b, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldNopReshape),
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
            return new Function(b, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldNopClamp),
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
            return new Function(e, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
    };

    public Dictionary<Var, IValue> FeedDict => new();
}

/// <summary>
/// transpose + pad + transpose => pad.
/// </summary>
public class FoldTransposePadCase : IRewriteCase
{
    private readonly Var _input;

    public FoldTransposePadCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 1, 2 }));
    }

    public Function PreExpr
    {
        get
        {
            var v0 = NHWCToNCHW(_input); // [1,2,3,1]
            var v1 = IR.F.NN.Pad(
                v0,
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                    { 2, 2 },
                    { 1, 1 },
                },
                PadMode.Constant,
                1.0f); // [1,2,7,3]
            var v2 = NCHWToNHWC(v1); // [1,7,3,2]
            var v3 = v2 * IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 3, 7, 3, 2 }).Evaluate().AsTensor(); // [3,7,3,2]
            return new Function(v3, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.CombineTransposePad),
        typeof(Passes.Rules.Neutral.FoldConstCall),
    };

    public Dictionary<Var, IValue> FeedDict => new(ReferenceEqualityComparer.Instance)
    {
        { _input, IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 3, 1, 2 }).Evaluate() },
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
            return new Function(e, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
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
            return new Function(e, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.FoldNopTranspose),
    };

    public Dictionary<Var, IValue> FeedDict => new();
}

public class FoldNopTransposeCase3 : FoldNopTransposeCase2
{
    public new IEnumerable<IRewriteRule> Rules => new IRewriteRule[]
    {
        new Passes.Rules.Neutral.FoldTwoTransposes(),
        new Passes.Rules.Neutral.FoldNopTranspose(),
        new Passes.Rules.Neutral.CombineBinaryTranspose(),
    };
}

public class ClassicDemo : IRewriteCase
{
    public Function PreExpr
    {
        get
        {
            var x = (Const)1234;
            return new Function(x * 2 / 2, System.Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.Xmul1),
        typeof(Passes.Rules.Neutral.ReassociateDiv),
        typeof(Passes.Rules.Neutral.ReassociateMul),

        // new Passes.Rules.Neutral.Reassociate(),
        typeof(Passes.Rules.Neutral.XDivX),
    };

    public Dictionary<Var, IValue> FeedDict => new();
}

public sealed class TransposeDemoCase : FoldNopTransposeCase3
{
    public new Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 28, 28, 3 }).Evaluate().AsTensor();
            var conv1 = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
            var lhs = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(conv1), 8, out_channels: 8, 3, 1));
            var rhs = conv1 + IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { new long[] { 1, 14, 14, 8 } }).Evaluate().AsTensor();
            return new Function(lhs + rhs, System.Array.Empty<Var>());
        }
    }
}

/// <summary>
/// this case from mobilenet v1.
/// </summary>
public class MobileNetV1TransposeCase : IRewriteCase
{
    private readonly Var _input;

    public MobileNetV1TransposeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 32, 112, 112 }));
    }

    public Function PreExpr
    {
        get
        {
            var v5 = Transpose(_input, new[] { 0, 2, 3, 1 }); // f32[1,112,112,32]
            var v6 = Transpose(v5, new[] { 0, 3, 1, 2 }); // f32[1,32,112,112]
            var v7 = Conv2D(
                v6,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 32, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,64,112,112]
            var v8 = Transpose(v7, new[] { 0, 2, 3, 1 }); // f32[1,112,112,64]
            var v9 = Pad(
                v8,
                new[,]
                {
                    { 0, 0 },
                    { 0, 1 },
                    { 0, 1 },
                    { 0, 0 },
                },
                PadMode.Constant,
                0.0f); // f32[1,113,113,64]
            var v10 = Transpose(v9, new[] { 0, 3, 1, 2 }); // f32[1,64,113,113]
            var v11 = Conv2D(
                v10,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 1, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 2, 2 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                64,
                new[] { 0.0f, 6.0f }); // f32[1,64,56,56]
            return new Function(v11, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.FoldNopTranspose),
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.CombineTransposePad),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, new[] { 1, 32, 112, 112 }).Evaluate() },
    };
}

/// <summary>
/// this case from pad with transpose.
/// </summary>
public class PadTransposeCase : IRewriteCase
{
    private readonly Var _input;

    public PadTransposeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 10, 10, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var v0 = Pad(
                _input,
                new[,]
                {
                    { 0, 0 },
                    { 2, 2 },
                    { 2, 2 },
                    { 0, 0 },
                },
                PadMode.Constant,
                0.0f); // f32[1,14,14,16]
            var v1 = Transpose(v0, new[] { 0, 3, 1, 2 }); // f32[1,16,14,14]
            var v2 = Conv2D(
                v1,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,64,12,12]
            var v3 = Pad(
                v2,
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                    { 0, 0 },
                    { 0, 0 },
                },
                PadMode.Constant,
                0.0f); // f32[1,64,12,12]
            return new Function(v3, new Var[] { _input });
        }
    }

    public virtual IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.CombinePadTranspose),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

/// <summary>
/// note we can't use pad+transpose and transpoe+pad in graph pass.
/// </summary>
public sealed class PadTransposeCaseEgraph : PadTransposeCase
{
    public override IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.CombinePadTranspose),
        typeof(Passes.Rules.Neutral.CombineTransposePad),
    };
}

public sealed class TransposeLeakyRelu : IRewriteCase
{
    private readonly Var _input;

    public TransposeLeakyRelu()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 16, 15, 20 }));
    }

    public Function PreExpr
    {
        get
        {
            var v5 = Transpose(_input, new[] { 0, 2, 3, 1 }); // f32[1,15,20,16]
            var v6 = LeakyRelu(v5, 0.1f); // f32[1,15,20,16]
            var v7 = Transpose(v6, new[] { 0, 3, 1, 2 }); // f32[1,16,15,20]
            var v8 = Conv2D(
                v7,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.FoldNopTranspose),
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.CombineTransposeActivations),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public class ActivationsTranspose : IRewriteCase
{
    public ActivationsTranspose()
    {
        Input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 16, 15, 20 }));
    }

    public virtual Function PreExpr
    {
        get
        {
            var v5 = Transpose(Input, new[] { 0, 2, 3, 1 }); // f32[1,15,20,16]
            var v6 = LeakyRelu(v5, 0.1f); // f32[1,15,20,16]
            var v7 = Transpose(v6, new[] { 0, 3, 1, 2 }); // f32[1,16,15,20]
            var v8 = Conv2D(
                v7,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,] { { 1, 1 }, { 1, 1 } },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { Input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.FoldNopTranspose),
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.CombineActivationsTranspose),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { Input, Normal(DataTypes.Float32, 0, 1, 1, Input.CheckedShape.ToValueArray()).Evaluate() },
    };

    protected Var Input { get; }
}

public sealed class ActivationsTranspose2 : ActivationsTranspose
{
    public override Function PreExpr
    {
        get
        {
            var v5 = Transpose(Input, new[] { 0, 2, 3, 1 }); // f32[1,15,20,16]
            var v6 = Relu(v5); // f32[1,15,20,16]
            var v7 = Transpose(v6, new[] { 0, 3, 1, 2 }); // f32[1,16,15,20]
            var v8 = Conv2D(
                v7,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { Input });
        }
    }
}

/// <summary>
/// note the prelu scope also need transpose.
/// </summary>
public sealed class ActivationsTransposePRelu : ActivationsTranspose
{
    public override Function PreExpr
    {
        get
        {
            var v5 = Transpose(Input, new[] { 0, 2, 3, 1 }); // f32[1,15,20,16]
            var v6 = PRelu(v5, Tensor.From(Enumerable.Repeat(0.2f, 16).ToArray(), new[] { 1, 1, 16 })); // f32[1,15,20,16]
            var v7 = Transpose(v6, new[] { 0, 3, 1, 2 }); // f32[1,16,15,20]
            var v8 = Conv2D(
                v7,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { Input });
        }
    }
}

public sealed class ActivationsTransposePRelu2 : ActivationsTranspose
{
    public override Function PreExpr
    {
        get
        {
            var v5 = Transpose(Input, new[] { 0, 2, 3, 1 }); // f32[1,15,20,16]
            var v6 = PRelu(v5, Tensor.From(Enumerable.Repeat(0.2f, 16).ToArray(), new[] { 1, 1, 1, 16 })); // f32[1,15,20,16]
            var v7 = Transpose(v6, new[] { 0, 3, 1, 2 }); // f32[1,16,15,20]
            var v8 = Conv2D(
                v7,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { Input });
        }
    }
}

/// <summary>
/// test for prelu slope is scalar.
/// </summary>
public sealed class ActivationsTransposePRelu3 : ActivationsTranspose
{
    public override Function PreExpr
    {
        get
        {
            var v5 = Transpose(Input, new[] { 0, 2, 3, 1 }); // f32[1,15,20,16]
            var v6 = PRelu(v5, Tensor.FromScalar(0.2f)); // f32[1,15,20,16]
            var v7 = Transpose(v6, new[] { 0, 3, 1, 2 }); // f32[1,16,15,20]
            var v8 = Conv2D(
                v7,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { Input });
        }
    }
}

public sealed class RemoveMarkerCaseEgraph : IRewriteCase
{
    private readonly Var _input;

    public RemoveMarkerCaseEgraph()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 16, 15, 20 }));
    }

    public Function PreExpr
    {
        get
        {
            var v5 = RangeOfMarker(Transpose(_input, new[] { 0, 2, 3, 1 }), new float[] { float.NegativeInfinity, float.PositiveInfinity }); // f32[1,15,20,16]
            var v6 = RangeOfMarker(Relu6(v5), new float[] { 0.0f, 6.0f }); // f32[1,15,20,16]
            var v7 = RangeOfMarker(Transpose(v6, new[] { 0, 3, 1, 2 }), new float[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            var v8 = RangeOfMarker(
                Conv2D(
                v7,
                RangeOfMarker(Normal(DataTypes.Float32, 0, 1, 1, new[] { 16, 16, 3, 3 }).Evaluate().AsTensor(), new float[] { -1.0f, 1.0f }),
                RangeOfMarker(Normal(DataTypes.Float32, 0, 1, 1, new[] { 16 }).Evaluate().AsTensor(), new float[] { -1.0f, 1.0f }),
                new[] { 1, 1 },
                new[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }),
                new float[] { 0.0f, 6.0f }); // f32[1,16,15,20]
            return new Function(v8, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Lower.RemoveMarker),
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.FoldNopTranspose),
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.CombineTransposeActivations),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class Conv2DPadsCase : IRewriteCase
{
    private readonly Var _input;

    public Conv2DPadsCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 16, 56, 56 }));
    }

    public Function PreExpr
    {
        get
        {
            var v12 = Conv2D(
                _input,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 96, 16, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 96 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,96,56,56]
            var v13 = Pad(
                v12,
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                    { 0, 1 },
                    { 0, 1 },
                },
                PadMode.Constant,
                0.0f); // f32[1,96,57,57]
            var v14 = Conv2D(
                v13,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 96, 1, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 96 }).Evaluate().AsTensor(),
                new[] { 2, 2 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                96,
                new[] { 0.0f, 6.0f }); // f32[1,96,28,28]
            return new Function(v14, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConv2DPads),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class ReduceWindow2DPadsCase : IRewriteCase
{
    private readonly Var _input;

    public ReduceWindow2DPadsCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 16, 56, 56 }));
    }

    public Function PreExpr
    {
        get
        {
            var v0 = Conv2D(
                _input,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 96, 16, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 96 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,96,56,56]
            var v1 = Pad(
                v0,
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                    { 0, 1 },
                    { 0, 1 },
                },
                PadMode.Constant,
                0.0f); // f32[1,96,57,57]
            var v2 = ReduceWindow2D(
                ReduceOp.Max,
                v1,
                0,
                new[] { 3, 3 },
                new[] { 2, 2 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                false,
                false); // f32[1,96,28,28]
            return new Function(v2, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldReduceWindow2DPads),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class MergeBinaryBeforeConv2DCase : IRewriteCase
{
    private readonly Var _inputLhs;
    private readonly Var _inputRhs;

    public MergeBinaryBeforeConv2DCase()
    {
        _inputLhs = new Var("_inputLhs", new TensorType(DataTypes.Float32, new[] { 1, 256, 56, 56 }));
        _inputRhs = new Var("_inputRhs", new TensorType(DataTypes.Float32, new[] { 1, 256, 56, 56 }));
    }

    public Function PreExpr
    {
        get
        {
            var v0 = _inputLhs + _inputRhs;
            var v1 = v0 * Normal(DataTypes.Float32, 0, 1, 1, new[] { 1, 256, 1, 1 }).Evaluate().AsTensor();
            var v2 = v1 + Normal(DataTypes.Float32, 0, 1, 1, new[] { 1, 256, 1, 1 }).Evaluate().AsTensor();
            var v3 = v2; // 1, 56, 56, 256
            var v4 = v3;
            var v5 = Conv2D(
                v4,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 256, 256, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 256 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                  { 0, 0 },
                  { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,256,56,56]
            var v6 = v5; // f32[1,256,56,56]
            var v7 = Pad(
                v6,
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                    { 1, 1 },
                    { 1, 1 },
                },
                PadMode.Constant,
                0.0f); // f32[1,256,58,58]
            var v8 = v7;
            var v9 = Conv2D(
                v8,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 256, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                  { 0, 0 },
                  { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,64,56,56]
            var v10 = v9; // f32[1,64,56,56]

            var v11 = v10;
            var v12 = Conv2D(
                v11,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 256, 64, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { 256 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                  { 0, 0 },
                  { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,256,56,56]
            var v13 = v12; // f32[1,256,56,56]
            var v16 = v0 + v13;

            return new Function(v16, new Var[] { _inputLhs, _inputRhs });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.FoldConv2DAddMul),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _inputLhs, Normal(DataTypes.Float32, 0, 1, 1, _inputLhs.CheckedShape.ToValueArray()).Evaluate() },
        { _inputRhs, Normal(DataTypes.Float32, 0, 1, 1, _inputRhs.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class CombineClampAddMul : IRewriteCase
{
    private const int _channels = 4; // 256
    private const int _featrueMap = 3; // 56
    private readonly Var _inputLhs;
    private readonly Var _inputRhs;

    public CombineClampAddMul()
    {
        _inputLhs = new Var("_inputLhs", new TensorType(DataTypes.Float32, new[] { 1, _featrueMap, _featrueMap, _channels }));
        _inputRhs = new Var("_inputRhs", new TensorType(DataTypes.Float32, new[] { 1, _featrueMap, _featrueMap, _channels }));
    }

    public Function PreExpr
    {
        get
        {
            var v0 = _inputLhs + _inputRhs; // [1,_featrueMap,_featrueMap,_channels]
            var v1 = v0 * Normal(DataTypes.Float32, 0, 1, 1, new[] { _channels }).Evaluate().AsTensor();
            var v2 = v1 + Normal(DataTypes.Float32, 0, 1, 2, new[] { _channels }).Evaluate().AsTensor();
            var v3 = Relu(v2);
            return new Function(v3, new Var[] { _inputLhs, _inputRhs });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
      typeof(Passes.Rules.Neutral.ReluToClamp),
      typeof(Passes.Rules.Neutral.CombineClampAdd),
      typeof(Passes.Rules.Neutral.CombineClampMul),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _inputLhs, Normal(DataTypes.Float32, 0, 1, 5, _inputLhs.CheckedShape.ToValueArray()).Evaluate() },
        { _inputRhs, Normal(DataTypes.Float32, 0, 1, 6, _inputRhs.CheckedShape.ToValueArray()).Evaluate() },
    };

    public bool ChecksPostCallBack(Function post)
    {
        return true;
    }
}

public sealed class FoldConv2DBnCase : IRewriteCase
{
    private const int _channels = 16; // 256
    private const int _featrueMap = 8; // 56
    private readonly Var _inputLhs;
    private readonly Var _inputRhs;

    public FoldConv2DBnCase()
    {
        _inputLhs = new Var("_inputLhs", new TensorType(DataTypes.Float32, new[] { 1, _featrueMap, _featrueMap, _channels }));
        _inputRhs = new Var("_inputRhs", new TensorType(DataTypes.Float32, new[] { 1, _featrueMap, _featrueMap, _channels }));
    }

    public Function PreExpr
    {
        get
        {
            var v0 = _inputLhs + _inputRhs; // [1,_featrueMap,_featrueMap,_channels]
            var v1 = v0 * Normal(DataTypes.Float32, 0, 1, 1, new[] { _channels }).Evaluate().AsTensor();
            var v2 = v1 + Normal(DataTypes.Float32, 0, 1, 2, new[] { _channels }).Evaluate().AsTensor();
            var v3 = Relu(v2);
            var v4 = NHWCToNCHW(v3);
            var v5 = Conv2D(
                v4,
                Normal(DataTypes.Float32, 0, 1, 3, new[] { _channels, _channels, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { _channels }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                  { 0, 0 },
                  { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,_channels,_featrueMap,_featrueMap]
            var v6 = NCHWToNHWC(v5); // f32[1,_featrueMap,_featrueMap,_channels]
            var v7 = Pad(
                v6,
                new[,]
                {
                    { 0, 0 },
                    { 1, 1 },
                    { 1, 1 },
                    { 0, 0 },
                },
                PadMode.Constant,
                0.0f); // f32[1,58,58,_channels]
            var v8 = NHWCToNCHW(v7);
            var v9 = Conv2D(
                v8,
                Normal(DataTypes.Float32, 0, 1, 4, new[] { 64, _channels, 3, 3 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 5, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                  { 0, 0 },
                  { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,64,_featrueMap,_featrueMap]
            var v10 = NCHWToNHWC(v9); // f32[1,_featrueMap,_featrueMap,64]
            var v11 = NHWCToNCHW(v10);
            var v12 = Conv2D(
                v11,
                Normal(DataTypes.Float32, 0, 1, 6, new[] { _channels, 64, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 7, new[] { _channels }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                  { 0, 0 },
                  { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f }); // f32[1,_channels,_featrueMap,_featrueMap]
            var v13 = NCHWToNHWC(v12); // f32[1,_featrueMap,_featrueMap,_channels]
            var v16 = v0 + v13;

            return new Function(v16, new Var[] { _inputLhs, _inputRhs });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
      typeof(Passes.Rules.Neutral.ReluToClamp),
      typeof(Passes.Rules.Neutral.CombineClampAdd),
      typeof(Passes.Rules.Neutral.CombineClampMul),
      typeof(Passes.Rules.Neutral.FoldTwoTransposes),
      typeof(Passes.Rules.Neutral.FoldNopTranspose),
      typeof(Passes.Rules.Neutral.CombineBinaryTranspose),
      typeof(Passes.Rules.Neutral.CombineTransposeActivations),
      typeof(Passes.Rules.Neutral.CombineConstBinaryTranspose),
      typeof(Passes.Rules.Neutral.CombineTransposeConstBinary),
      typeof(Passes.Rules.Neutral.FoldConv2DAddMul),
      typeof(Passes.Rules.Neutral.FoldConstCall),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _inputLhs, Normal(DataTypes.Float32, 0, 1, 1, _inputLhs.CheckedShape.ToValueArray()).Evaluate() },
        { _inputRhs, Normal(DataTypes.Float32, 0, 1, 1, _inputRhs.CheckedShape.ToValueArray()).Evaluate() },
    };

    public bool ChecksPostCallBack(Function post)
    {
        return true;
    }
}

public sealed class FoldLayerNormCase : IRewriteCase
{
    private readonly Var _input;

    public FoldLayerNormCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new[] { 1, 3, 16, 16 };
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
            long[] axes = { 0 };
            float initValue = 0F;
            long keepDims = 1;
            var v0 = input;
            var v1 = IR.F.Tensors.Reshape(v0, shape);
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v1, axes, initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v1, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Pow, v3, 1f);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, axes, initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Sqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Div, v3, v7);
            var v9 = IR.F.Tensors.Reshape(v8, shape);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v9, 1f);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, 1f);
            var rootPre = v11;
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldLayerNormPattern1),
        typeof(Passes.Rules.Neutral.FoldLayerNormPattern2),
        typeof(Passes.Rules.Neutral.FoldLayerNormPattern3),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class FoldSwishCase : IRewriteCase
{
    private readonly Var _input;

    public FoldSwishCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new[] { 1, 3, 16, 16 };
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
            var v0 = input;
            var v1 = IR.F.NN.Sigmoid(v0);
            var v2 = IR.F.Math.Binary(BinaryOp.Mul, v1, v0);
            var rootPre = v2;
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldSwishPattern1),
        typeof(Passes.Rules.Neutral.FoldSwishPattern2),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class FoldGeluCase : IRewriteCase
{
    private readonly Var _input;

    public FoldGeluCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, new[] { 1, 3, 16, 16 });
            var v0 = input;
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v0, 0.577350f); // "mul3Call"
            var v1 = IR.F.Math.Binary(BinaryOp.Div, v4, 1.414213f); // divCall
            var v2 = IR.F.NN.Erf(v1); // "erfCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Add, v2, 1f); // "addCall"
            var v5 = IR.F.Math.Binary(BinaryOp.Mul, v4, v3); // "mul2Call"
            var v6 = IR.F.Math.Binary(BinaryOp.Mul, v5, 0.5f); // "Mul1Call"
            var rootPre = v6;
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldGeneralGelu),
        typeof(Passes.Rules.Neutral.FoldGeluWithScale),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class FoldHardSwishCase : IRewriteCase
{
    private readonly Var _input;

    public FoldHardSwishCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, new[] { 1, 3, 16, 16 });
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Add, v0, 3f); // "addCall"
            var v2 = IR.F.Math.Clamp(v1, new ValueRange<float>(0f, 6f)); // "clampCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Mul, v2, v0); // "mulCall"
            var v4 = IR.F.Math.Binary(BinaryOp.Div, v3, 6f); // "divCall"
            var rootPre = v4;
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldHardSwish),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class MatMulToConv2DCase : IRewriteCase
{
    private readonly Var _inputLhs;
    private readonly Var _inputRhs;

    public MatMulToConv2DCase()
    {
        _inputLhs = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 5 }));
        _inputRhs = new Var("input", new TensorType(DataTypes.Float32, new[] { 5, 1 }));
    }

    public Function PreExpr
    {
        get
        {
            var a = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 5 });
            var b = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 5, 1 }).Evaluate();
            var rootPre = Math.MatMul(a, b.AsTensor());
            return new Function(rootPre, new Var[] { _inputLhs, _inputRhs });
        }
    }

    public IEnumerable<Type> Rules { get; } = new Type[]
    {
        typeof(Passes.Rules.Neutral.MatMulToConv2D),
    };

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _inputLhs, Normal(DataTypes.Float32, 0, 1, 1, _inputLhs.CheckedShape.ToValueArray()).Evaluate() },
        { _inputRhs, Normal(DataTypes.Float32, 0, 1, 1, _inputRhs.CheckedShape.ToValueArray()).Evaluate() },
    };
}
