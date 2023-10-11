// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Transforms;
using Nncase.Schedule;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tuple = Nncase.IR.Tuple;

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

public sealed class MultiReshapeCase : IRewriteCase
{
    private readonly int _n = 1;
    private readonly int _ic = 4;
    private readonly int _h = 60;
    private readonly int _w = 72;
    private readonly int _oc = 1;

    private readonly Var _input;

    public MultiReshapeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { _n, _ic, _h, _w }));
        FeedDict = new Dictionary<Var, IValue>() { { _input, Normal(DataTypes.Float32, 1, 1, 1, new[] { _n, _ic, _h, _w }).Evaluate() } };
    }

    public Function PreExpr
    {
        get
        {
            var conv = Conv2D(
                _input,
                Normal(DataTypes.Float32, 0, 1, 1, new[] { _oc, _ic, 1, 1 }).Evaluate().AsTensor(),
                Normal(DataTypes.Float32, 0, 1, 1, new[] { _oc }).Evaluate().AsTensor(),
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
            var x = conv;
            x = Reshape(x, new[] { _n, _oc * _h * _w });
            x = Reshape(x, new[] { _n, _oc, _h * _w });
            x = Reshape(x, new[] { _n, _oc * _h, _w });
            x = Reshape(x, new[] { _n * _oc, _h * _w });
            x = Reshape(x, new[] { _n * _oc, _h, _w });
            x = Reshape(x, new[] { _n * _oc * _h, _w });
            x = Reshape(x, new[] { -1, _w });
            return new Function(x, Array.Empty<Var>());
        }
    }

    public IEnumerable<Type> Rules => new Type[]
    {
        typeof(Passes.Rules.Neutral.FoldConstCall),
        typeof(Passes.Rules.Neutral.FoldNopTranspose),
        typeof(Passes.Rules.Neutral.FoldTwoTransposes),
        typeof(Passes.Rules.Neutral.CombineTransposeUnary),
        typeof(Passes.Rules.Neutral.CombineTransposePad),
        typeof(Passes.Rules.Neutral.CombinePadTranspose),
        typeof(Passes.Rules.Neutral.CombineBinaryTranspose),
        typeof(Passes.Rules.Neutral.CombineConstBinaryTranspose),
        typeof(Passes.Rules.Neutral.CombineTransposeConstBinary),
        typeof(Passes.Rules.Neutral.CombineTransposeReduce),
        typeof(Passes.Rules.Neutral.CombineTransposeActivations),
        typeof(Passes.Rules.Neutral.CombineActivationsTranspose),
        typeof(Passes.Rules.Neutral.CombineTransposeConcat),
        typeof(Passes.Rules.Neutral.CombineBinaryReshape),
        typeof(Passes.Rules.Neutral.CombineConstBinaryReshape),
        typeof(Passes.Rules.Neutral.CombineUnaryReshape),
        typeof(Passes.Rules.Neutral.CombineActivationsReshape),
        typeof(Passes.Rules.Neutral.CombineReshapePad),
        typeof(Passes.Rules.Neutral.FoldNopPad),
        typeof(Passes.Rules.Neutral.FoldConv2DPads),
        typeof(Passes.Rules.Neutral.FuseClampConv2D),
        typeof(Passes.Rules.Neutral.FoldReduceWindow2DPads),
        typeof(Passes.Rules.Neutral.SqueezeToReshape),
        typeof(Passes.Rules.Neutral.UnSqueezeToReshape),
        typeof(Passes.Rules.Neutral.TransposeToReshape),
        typeof(Passes.Rules.Neutral.FlattenToReshape),
        typeof(Passes.Rules.Neutral.ReshapeToTranspose),
        typeof(Passes.Rules.Neutral.FoldNopReshape),
        typeof(Passes.Rules.Neutral.FoldTwoReshapes),
        typeof(Passes.Rules.Neutral.ReluToClamp),
        typeof(Passes.Rules.Neutral.Relu6ToClamp),
        typeof(Passes.Rules.Neutral.FoldNopSlice),
        typeof(Passes.Rules.Neutral.FoldTwoSlices),
    };

    public Dictionary<Var, IValue> FeedDict { get; }
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
            var rhs = conv1 + IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 14, 14, 8 }).Evaluate().AsTensor();
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
        typeof(Passes.Rules.Neutral.FoldHardSwish1),
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

public sealed class ReduceCase : IRewriteCase
{
    private readonly Var _input;

    public ReduceCase()
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
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v0, axes, initValue, keepDims);
            return new Function(v5, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class BroadcastCase : IRewriteCase
{
    private readonly Var _input;

    public BroadcastCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            _ = new long[] { 16 };
            var newShape = new[] { 1, 3, 16, 16 };
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, new[] { 1, 3, 16, 16 });
            var expr = IR.F.Tensors.Broadcast(input, newShape);
            var rootPre = expr;
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class CastCase : IRewriteCase
{
    private readonly Var _input;

    public CastCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, new[] { 1, 3, 16, 16 });
            var expr = IR.F.Tensors.Cast(input, DataTypes.Int32);
            var rootPre = expr;
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class TileCase : IRewriteCase
{
    private readonly Var _input;

    public TileCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, new[] { 1, 3, 16, 16 });
            var expr = IR.F.Tensors.Tile(input, new[] { 1L, 1L, 1L, 1L });
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class StackCase : IRewriteCase
{
    private readonly Var _input;

    public StackCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1 }));
    }

    public Function PreExpr
    {
        get
        {
            Expr a = 1;
            Expr b = 2;
            var inputList = new Tuple(a, b);
            var expr = IR.F.Tensors.Stack(inputList, 0);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class BitcastCase : IRewriteCase
{
    private readonly Var _input;

    public BitcastCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, new[] { 1, 3, 16, 16 });
            var expr = IR.F.Tensors.Bitcast(DataTypes.Float32, input, DataTypes.Float32, new[] { 1, 3, 32, 8 });
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SliceCase : IRewriteCase
{
    private readonly Var _input;

    public SliceCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 3, 4, 5 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = Tensor.From<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
            var begin = Tensor.From<int>(new[] { 0, 0, 0, 0 }, new Shape(new[] { 4 }));
            var end = Tensor.From<int>(new[] { 1, 1, 1, 5 }, new Shape(new[] { 4 }));
            var axes = Tensor.From<int>(new[] { 0, 1, 2, 3 }, new Shape(new[] { 4 }));
            var strides = Tensor.From<int>(new[] { 1, 1, 1, 1 }, new Shape(new[] { 4 }));
            _ = Const.FromTensor(Tensor.From<int>(Enumerable.Range(0, 5).ToArray(), new Shape(new[] { 1, 1, 1, 5 })));
            var expr = IR.F.Tensors.Slice(input, begin, end, axes, strides);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class LRNCase : IRewriteCase
{
    private readonly Var _input;

    public LRNCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 1, 3, 16, 16 };
            var input = OrtKI.Random(shape);
            var alpha = 0.001F;
            var beta = 0.5F;
            var bias = 0.8F;
            var size = 3L;
            var expr = IR.F.NN.LRN(input.ToTensor(), alpha, beta, bias, size);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SoftmaxCase : IRewriteCase
{
    private readonly Var _input;

    public SoftmaxCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var ortTensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var nncaseTensor = ortTensor.ToTensor();
            var expr = IR.F.NN.Softmax(nncaseTensor, -1L);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class CumSumCase : IRewriteCase
{
    private readonly Var _input;

    public CumSumCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var axis = 0;
            var exclusive = false;
            var reverse = false;

            var input1 = Tensor.From(input, new[] { 2, 4 });
            var expr = IR.F.Tensors.CumSum(input1, axis, exclusive, reverse);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class LSTMCase : IRewriteCase
{
    private readonly Var _input;

    public LSTMCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 1, 2 }));
    }

    public Function PreExpr
    {
        get
        {
            var inputSize = 2;
            var hiddenSize = 1;
            var outputSize = 1;
            var direction = LSTMDirection.Forward;
            var batchSize = 1;
            var seqLength = 1;
            var numberDirections = 1;
            var x = OrtKI.Random(seqLength, batchSize, inputSize);
            var initC = OrtKI.Random(numberDirections, batchSize, hiddenSize);
            var initH = OrtKI.Random(numberDirections, batchSize, hiddenSize);
            var b = OrtKI.Random(numberDirections, 8 * hiddenSize);
            var w = OrtKI.Random(numberDirections, 4 * hiddenSize, inputSize);
            var r = OrtKI.Random(numberDirections, 4 * hiddenSize, hiddenSize);
            var p = new float[numberDirections, 3 * hiddenSize];
            var acts = new[] { "Sigmoid", "Tanh", "Tanh" };
            var expr = IR.F.RNN.LSTM(direction, LSTMLayout.Zero, acts, x.ToTensor(), w.ToTensor(), r.ToTensor(), b.ToTensor(), new[] { seqLength }, initH.ToTensor(), initC.ToTensor(), p, 0, 0, float.NaN, hiddenSize, 0, outputSize);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class InstanceNormalizationCase : IRewriteCase
{
    private readonly Var _input;

    public InstanceNormalizationCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 1, 3, 16, 16 };
            var x = OrtKI.Random(shape);
            var scale = OrtKI.Random(new long[] { shape[1] });
            var b = OrtKI.Random(new long[] { shape[1] });
            var epsilon = 0.01F;
            var expr = IR.F.NN.InstanceNormalization(x.ToTensor(), scale.ToTensor(), b.ToTensor(), epsilon);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class HardSwishCase : IRewriteCase
{
    private readonly Var _input;

    public HardSwishCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var expr = IR.F.NN.HardSwish(input.ToTensor());
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SoftplusCase : IRewriteCase
{
    private readonly Var _input;

    public SoftplusCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var expr = IR.F.NN.Softplus(input.ToTensor());
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SoftsignCase : IRewriteCase
{
    private readonly Var _input;

    public SoftsignCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var expr = IR.F.NN.Softsign(input.ToTensor());
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class LpNormalizationCase : IRewriteCase
{
    private readonly Var _input;

    public LpNormalizationCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var expr = IR.F.NN.LpNormalization(input.ToTensor(), 0L, 1L);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class Conv2DTransposeCase : IRewriteCase
{
    private readonly Var _input;

    public Conv2DTransposeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 1, 5, 5 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(1, 1, 5, 5);
            var weight = OrtKI.Random(2, 1, 3, 3);
            var bias = OrtKI.Random(2);
            var outShape = Tensor.From(new long[] { 1, 2, 5, 5 }, new Shape(4));
            var expr = IR.F.NN.Conv2DTranspose(
                input.ToTensor(),
                weight.ToTensor(),
                bias.ToTensor(),
                outShape,
                stride: new[] { 1, 1 },
                padding: Tensor.From<long>(new long[] { 1, 1, 1, 1 }, new[] { 4 }),
                outputPadding: Tensor.From<long>(new long[] { 0, 0 }, new[] { 2 }),
                dilation: new[] { 1, 1 },
                PadMode.Constant,
                1);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class LogSoftmaxCase : IRewriteCase
{
    private readonly Var _input;

    public LogSoftmaxCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var ortTensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var nncaseTensor = ortTensor.ToTensor();
            var expr = IR.F.NN.LogSoftmax(nncaseTensor, -1L);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class CompareCase : IRewriteCase
{
    private readonly Var _input;

    public CompareCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 10 }));
    }

    public Function PreExpr
    {
        get
        {
            var expr_a = Tensor.FromScalar<int>(10);
            var expr = IR.F.Math.Compare(CompareOp.Equal, expr_a, expr_a);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class FakeDequantizeCase : IRewriteCase
{
    private readonly Var _input;

    public FakeDequantizeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = new byte[] { 127, 128, 150, 160, 170, 180, 200, 205 };
            byte zero_point = 127;
            var scale = 0.01F;
            var expr = IR.F.Math.FakeDequantize(
                Tensor.From(input, new[] { 2, 4 }),
                new QuantParam(zero_point, scale),
                DataTypes.Float32);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class FakeQuantizeCase : IRewriteCase
{
    private readonly Var _input;

    public FakeQuantizeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
            byte zero_point = 127;
            var scale = 0.05F;
            var expr = IR.F.Math.FakeQuantize(
                Tensor.From(input, new[] { 2, 4 }),
                new QuantParam(zero_point, scale),
                DataTypes.UInt8);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class TopKCase : IRewriteCase
{
    private readonly Var _input;

    public TopKCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 3, 4, 5 }));
    }

    public Function PreExpr
    {
        get
        {
            var x = Tensor.From<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
            var k = 1L;
            var axis = -1;
            var largest = 1;
            var sorted = 1;
            var expr = IR.F.Tensors.TopK(x, k, axis, largest, sorted);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class GatherCase : IRewriteCase
{
    private readonly Var _input;

    public GatherCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 0, 1, 2, 3 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new[] { 2, 2 };
            var input = new Tensor<int>(new[] { 0, 1, 2, 3 }, shape);
            var indices = new Tensor<long>(new[] { 0L, 0L, 1L, 1L }, shape);
            long batchDims = 0L;
            var expr = IR.F.Tensors.Gather(input, (int)batchDims, indices);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class GatherNDCase : IRewriteCase
{
    private readonly Var _input;

    public GatherNDCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 0, 1, 2, 3 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new[] { 2, 2 };
            var input = new Tensor<int>(new[] { 0, 1, 2, 3 }, shape);
            var indices = new Tensor<long>(new[] { 0L, 0L, 1L, 1L }, shape);
            long batchDims = 0L;
            var expr = IR.F.Tensors.GatherND(input, batchDims, indices);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class FlattenCase : IRewriteCase
{
    private readonly Var _input;

    public FlattenCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 1, 3, 16, 16 };
            var input = OrtKI.Random(shape);
            var expr = IR.F.Tensors.Flatten(input.ToTensor(), -1);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SplitCase : IRewriteCase
{
    private readonly Var _input;

    public SplitCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 1, 3, 16, 16 };
            var input = OrtKI.Random(shape);
            var axis = 1L;
            var sections = new long[] { 1, 2 };
            var expr = IR.F.Tensors.Split(input.ToTensor(), axis, sections);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SqueezeCase : IRewriteCase
{
    private readonly Var _input;

    public SqueezeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 1, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 1, 3, 1, 16 };
            var input = OrtKI.Random(shape);
            var axes = new long[] { 0, 2 };
            var expr = IR.F.Tensors.Squeeze(input.ToTensor(), axes);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ConcatCase : IRewriteCase
{
    private readonly Var _input;

    public ConcatCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var a = Const.FromTensor(Tensor.From<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 })));
            var b = Const.FromTensor(Tensor.From<int>(new int[12], new Shape(new[] { 1, 3, 4 })));
            var inputList = new Tuple(a, b);
            var expr = IR.F.Tensors.Concat(inputList, 0);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class UnsqueezeCase : IRewriteCase
{
    private readonly Var _input;

    public UnsqueezeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 6 }));
    }

    public Function PreExpr
    {
        get
        {
            var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 4, 6 });
            var rootPre = IR.F.Tensors.Unsqueeze(a, new[] { 0, 2 });
            return new Function(rootPre, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ExpandCase : IRewriteCase
{
    private readonly Var _input;

    public ExpandCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var oldShape = new long[] { 1, 16 };
            var input = OrtKI.Random(oldShape);
            var expr = IR.F.Tensors.Expand(input.ToTensor(), new long[] { 16, 16 });
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ShapeOfCase : IRewriteCase
{
    private readonly Var _input;

    public ShapeOfCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2, 3 }));
    }

    public Function PreExpr
    {
        get
        {
            var v = Tensor.From<int>(new[] { 1, 2, 3 });
            var shape = ShapeOf(v);
            return new Function(shape, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ReverseSequenceCase : IRewriteCase
{
    private readonly Var _input;

    public ReverseSequenceCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 4, 4 };
            var input = OrtKI.Random(shape);
            var seqLens = Tensor.From<long>(new long[] { 1, 2, 3, 4 });
            var batchAxis = 1L;
            var timeAxis = 0L;
            var expr = IR.F.Tensors.ReverseSequence(input.ToTensor(), seqLens, batchAxis, timeAxis);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class WhereCase : IRewriteCase
{
    private readonly Var _input;

    public WhereCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 2, 2 }));
    }

    public Function PreExpr
    {
        get
        {
            var shape = new long[] { 2, 2 };
            var con = new Tensor<bool>(new[] { true, false, true, true }, new[] { 2, 2 });
            var x = OrtKI.Random(shape);
            var y = OrtKI.Random(shape);
            var expr = IR.F.Tensors.Where(con, x.ToTensor(), y.ToTensor());
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class RangeCase : IRewriteCase
{
    private readonly Var _input;

    public RangeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var begin = 0F;
            var end = 100F;
            var step = 2F;
            var expr = IR.F.Tensors.Range(begin, end, step);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SizeOfCase : IRewriteCase
{
    private readonly Var _input;

    public SizeOfCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            _ = new Shape(new[] { 1, 3, 16, 16 });
            var input = OrtKI.Random(1, 3, 16, 16).ToTensor();
            var expr = IR.F.Tensors.SizeOf(input);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class BatchToSpaceCase : IRewriteCase
{
    private readonly Var _input;

    public BatchToSpaceCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4, 1, 2, 2 }));
    }

    public Function PreExpr
    {
        get
        {
            var a = new float[] { 1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16 };
            var input = Tensor.From(a, new[] { 4, 1, 2, 2 });
            var shape = new long[] { 2, 2 };
            _ = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            var crops = new long[] { 0, 0, 0, 0 };
            var expr = IR.F.NN.BatchToSpace(
                input,
                Tensor.From(shape, new[] { 2 }),
                Tensor.From(crops, new[] { 2, 2 }));
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class L2NormalizationCase : IRewriteCase
{
    private readonly Var _input;

    public L2NormalizationCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1 }));
    }

    public Function PreExpr
    {
        get
        {
            var a = new float[] { 0F, 2F, 3F, 2F, 2F, 2F };
            _ = new float[] { 0F, 0.4F, 0.6F, 0.4F, 0.4F, 0.4F };
            var input = Tensor.From(a, new[] { 6 });
            var expr = IR.F.NN.L2Normalization(input);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class OneHotCase : IRewriteCase
{
    private readonly Var _input;

    public OneHotCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var a = new int[] { 1, 2, 0, 3 };
            var indices = Tensor.From(a, new[] { 4 });
            var depth = 5;
            var values = Tensor.From(new int[] { 0, 1 }, new Shape(new[] { 2 }));
            var axis = 0L;
            var expr = IR.F.NN.OneHot(OneHotMode.Normal, indices, depth, values, axis);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class CeluCase : IRewriteCase
{
    private readonly Var _input;

    public CeluCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var alpha = 0.8F;
            var expr = IR.F.NN.Celu(input.ToTensor(), alpha);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class EluCase : IRewriteCase
{
    private readonly Var _input;

    public EluCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var alpha = 0.8F;
            var expr = IR.F.NN.Elu(input.ToTensor(), alpha);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class SeluCase : IRewriteCase
{
    private readonly Var _input;

    public SeluCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var alpha = 1.2F;
            var gamma = 1.3F;
            var expr = IR.F.NN.Selu(input.ToTensor(), alpha, gamma);
            return new Function(expr, _input);
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class HardmaxCase : IRewriteCase
{
    private readonly Var _input;

    public HardmaxCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var ortTensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var nncaseTensor = ortTensor.ToTensor();
            var expr = IR.F.NN.Hardmax(nncaseTensor, -1L);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class HardSigmoidCase : IRewriteCase
{
    private readonly Var _input;

    public HardSigmoidCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
            var alpha = 1.2F;
            var gamma = 1.3F;
            var expr = IR.F.NN.HardSigmoid(input.ToTensor(), alpha, gamma);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ReduceArgCase : IRewriteCase
{
    private readonly Var _input;

    public ReduceArgCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1 }));
    }

    public Function PreExpr
    {
        get
        {
            long axis = 0L;
            long select_last_idx = 0L;
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var result = new int[] { 5, 6, 7, 8 };
            var expr_a = Tensor.From(a, new[] { 2, 4 });
            _ = Tensor.From(result, new[] { 1, 4 }).ToOrtTensor();
            var expr = IR.F.Tensors.ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, expr_a, axis, 0L, select_last_idx);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class NormalLikeCase : IRewriteCase
{
    private readonly Var _input;

    public NormalLikeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var mean = 0.5F;
            var scale = 1F;
            var seed = 1F;
            var shape = new long[] { 1, 3, 16, 16 };
            var input = OrtKISharp.Tensor.Empty(shape);
            var expr = IR.F.Random.NormalLike(DataTypes.Float32, input.ToTensor(), mean, scale, seed);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class UniformLikeCase : IRewriteCase
{
    private readonly Var _input;

    public UniformLikeCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var high = 1F;
            var low = 0F;
            var seed = 1F;
            var shape = new long[] { 1, 3, 16, 16 };
            var input = OrtKISharp.Tensor.Empty(shape);
            var expr = IR.F.Random.UniformLike(DataTypes.Float32, input.ToTensor(), high, low, seed);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class UniformCase : IRewriteCase
{
    private readonly Var _input;

    public UniformCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 16, 16 }));
    }

    public Function PreExpr
    {
        get
        {
            var high = 1F;
            var low = 0F;
            var seed = 1F;
            var shape = new long[] { 1, 3, 16, 16 };
            var expr = IR.F.Random.Uniform(DataTypes.Float32, high, low, seed, shape);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ResizeImageCase : IRewriteCase
{
    private readonly Var _input;

    public ResizeImageCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = OrtKI.Random(1, 3, 224, 224).ToTensor();
            var expr = IR.F.Imaging.ResizeImage(ImageResizeMode.Bilinear, input, Array.Empty<int>(), new[] { 1, 3, 112, 112 }, isTFResize: true);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
     {
         { _input, Normal(DataTypes.Float32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
     };
}

public sealed class ProdCase : IRewriteCase
{
    private readonly Var _input;

    public ProdCase()
    {
        _input = new Var("input", new TensorType(DataTypes.Int32, new[] { 1, 2, 3, 4 }));
    }

    public Function PreExpr
    {
        get
        {
            var input = Tensor.From<int>(new[] { 1, 2, 3, 4 });
            var expr = Tensors.Prod(input);
            return new Function(expr, new Var[] { _input });
        }
    }

    public IEnumerable<Type> Rules { get; } = Array.Empty<Type>();

    public Dictionary<Var, IValue> FeedDict => new()
    {
        { _input, Normal(DataTypes.Int32, 0, 1, 1, _input.CheckedShape.ToValueArray()).Evaluate() },
    };
}

public sealed class PReluTransposeCase : IRewriteCase
{
    public PReluTransposeCase()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 33, 65, 1 }));
        {
            var v0 = Transpose(input, new[] { 0, 3, 1, 2 }); // f32[1,1,33,65]
            var v1 = IR.F.NN.Conv2D(v0, IR.F.Random.Normal(new[] { 8, 1, 3, 3 }).Evaluate().AsTensor(), new[] { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f }, new[] { 1, 1 }, new[,] { { 1, 1 }, { 1, 1 } }, new[] { 1, 1 }, PadMode.Constant, 1, new[] { -float.PositiveInfinity, float.PositiveInfinity }); // f32[1,8,33,65]
            var v2 = Transpose(v1, new[] { 0, 2, 3, 1 }); // f32[1,33,65,8]
            var v3 = PRelu(v2, Tensor.From(new[] { -0.12399824f, -0.03634571f, 0.5353417f, -0.67039806f, 0.91027457f, -1.0752988f, 0.55657554f, -1.1045103f }, new[] { 1, 1, 8 })); // f32[1,33,65,8]
            PreExpr = new Function(v3, new[] { input });
        }

        FeedDict = new() { { input, IR.F.Random.Normal(new[] { 1, 33, 65, 1 }).Evaluate() } };
    }

    public Function PreExpr { get; }

    public IEnumerable<Type> Rules => new[] {
        typeof(CombineTransposeActivations),
        typeof(CombineActivationsTranspose),
        typeof(TransposeToReshape),
        typeof(ReshapeToTranspose),
        typeof(FoldNopTranspose),
        typeof(FoldNopReshape),
    };

    public Dictionary<Var, IValue> FeedDict { get; }
}

public sealed class ReshapeTransposeReshapeCase : IRewriteCase
{
    public ReshapeTransposeReshapeCase()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 77, 768 }));
        {
            var v0 = Reshape(input, new[] { 1, 77, 12, 64 });
            var v2 = Transpose(v0, new[] { 0, 2, 1, 3 });
            var v3 = Reshape(v2, new[] { 12, 77, 64 });
            PreExpr = new Function(v3, new[] { input });
        }

        FeedDict = new() { { input, IR.F.Random.Normal(new[] { 1, 77, 768 }).Evaluate() } };
    }

    public Function PreExpr { get; }

    public IEnumerable<Type> Rules => new[] {
        typeof(CombineReshapeTranspose),
        typeof(FoldTwoReshapes),
    };

    public Dictionary<Var, IValue> FeedDict { get; }
}

public sealed class ReshapeBinaryConstReshapeCase : IRewriteCase
{
    public ReshapeBinaryConstReshapeCase()
    {
        var v9 = new Var("v9", new TensorType(DataTypes.Float32, new[] { 12, 77, 77 }));
        {
            var v10 = Reshape(v9, new[] { 1, 12, 77, 77 }); // f32[1,12,77,77]
            var v11 = IR.F.Math.Add(v10, IR.F.Random.Normal(new[] { 1, 1, 77, 77 }).Evaluate().AsTensor()); // f32[1,12,77,77]
            var v12 = Reshape(v11, new[] { 12, 77, 77 }); // f32[12,77,77]

            PreExpr = new Function(v12, new[] { v9 });
        }

        FeedDict = new() { { v9, IR.F.Random.Normal(new[] { 12, 77, 77 }).Evaluate() } };
    }

    public Function PreExpr { get; }

    public IEnumerable<Type> Rules => new[] {
        typeof(FoldReshapeBinaryConstReshape),
    };

    public Dictionary<Var, IValue> FeedDict { get; }
}
