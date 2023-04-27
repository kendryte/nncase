// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Nncase.Tests.TransformTest;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules;

[AutoSetupTestMethod(InitSession = true)]
public class ShapeBucketTest : TransformTestBase
{
    // [Fact]
    // public void TestToFusion()
    // {
    //     var lhs = Testing.Rand<float>(3, 1, 24, 24);
    //     var rhs = Testing.Rand<float>(3, 1, 24, 24);
    //     var tr = Transpose(lhs, new[] { 1, 0, 2, 3 });
    //     var abs = Abs(rhs);
    //     var matmul = IR.F.Math.MatMul(tr, abs);
    //     var expr = Abs(matmul);
    //     TestMatched<MatmulToFusion>(expr);
    // }
    [Fact]
    public void TestBucket()
    {
        CompileOptions.DumpFlags = DumpFlags.Rewrite;
        var dim = 27;
        var inputA = Testing.Rand<float>(1, 3, 24, dim);
        var inputB = Testing.Rand<float>(1, 3, dim, 24);
        var effectVar = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var lhs = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, 24, Dimension.Unknown }));
        var rhs = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 24 }));
        var f = new VarFusion("stackvm", effectVar, IR.F.Math.MatMul(lhs, rhs), lhs, rhs);
        var inputInfo = new Dictionary<Var, Expr[]>
        {
            { lhs, new[] { 1, 3, 24, (Expr)effectVar } }, { rhs, new[] { 1, 3, (Expr)effectVar, 24 } },
        };
        var call = new Call(f, inputA, inputB);
        Assert.True(call.InferenceType());
        TestMatchedCore(call, null, new[] { new FusionBucket(inputInfo) });
    }

    [Fact]
    public void TestFindVar()
    {
        var v1 = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var v2 = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var expr = ((v1 * 4) + (v2 * 3)) * 2;
        var visitor = new FindVar();
        visitor.Visit(expr);
        Assert.Equal(visitor.Vars, new HashSet<Var>(new[] { v1, v2 }));
    }

    [Fact]
    public void TestGetBoundDict()
    {
        var batch = new Var("batch", new TensorType(DataTypes.Int32, Shape.Scalar));
        var tok_len = new Var("tok_len", new TensorType(DataTypes.Int32, Shape.Scalar));
        var enc_len = new Var("enc_len", new TensorType(DataTypes.Int32, Shape.Scalar));
        var dec_len = new Var("dec_len", new TensorType(DataTypes.Int32, Shape.Scalar));
        var tokens = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, Dimension.Unknown }));
        var enc_k = new Var(new TensorType(DataTypes.Float32, new[] { 3, Dimension.Unknown, 1, 256 }));
        var enc_v = new Var(new TensorType(DataTypes.Float32, new[] { 3, Dimension.Unknown, 1, 256 }));
        var enc_pad_mask = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown }));
        var dec_k1 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_k2 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_k3 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_v1 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_v2 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_v3 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var inputInfo = new Dictionary<Var, Expr[]>
        {
            { tokens, new Expr[] { batch, tok_len } },
            { enc_k, new Expr[] { 3, enc_len, 1, 256 } },
            { enc_v, new Expr[] { 3, enc_len, 1, 256 } },
            { enc_pad_mask, new Expr[] { 1, enc_len } },
            { dec_k1, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_k2, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_k3, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_v1, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_v2, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_v3, new Expr[] { batch, 4, dec_len, 64 } },
        };

        // todo: import相同名字的var归一化
        var dict = new Dictionary<string, (int, int)>
        {
            { "batch", (24, 48) }, { "tok_len", (24, 48) }, { "enc_len", (24, 48) }, { "dec_len", (24, 48) },
        };
        var (minDict, maxDict) = ReplaceRewrite.GetBoundDict(inputInfo, dict);
        Assert.Equal(Enumerable.Repeat(48, minDict.Count).ToArray(), maxDict.Values.Select(x => x.AsTensor().ToScalar<int>()));
        Assert.Equal(Enumerable.Repeat(24, maxDict.Count).ToArray(), minDict.Values.Select(x => x.AsTensor().ToScalar<int>()));
    }

    public virtual Tensor[] MakeInputs(TensorType[] types)
    {
        // return types.Select(type => Testing.Rand(type.DType, type.Shape.ToValueArray())).ToArray();
        var batch = 1;
        var tok_len = 1;
        var enc_len = 1;
        var dec_len = 1;
        var in0 = Testing.Rand<long>(batch, tok_len);
        var in1 = Testing.Rand<float>(3, enc_len, 1, 256);
        var in2 = Testing.Rand<float>(3, enc_len, 1, 256);
        var in3 = Testing.Rand<bool>(1, enc_len);
        var in4 = Testing.Rand<float>(batch, 4, dec_len, 64);
        var in5 = Testing.Rand<float>(batch, 4, dec_len, 64);
        var in6 = Testing.Rand<float>(batch, 4, dec_len, 64);
        var in7 = Testing.Rand<float>(batch, 4, dec_len, 64);
        var in8 = Testing.Rand<float>(batch, 4, dec_len, 64);
        var in9 = Testing.Rand<float>(batch, 4, dec_len, 64);
        return new[] { (Tensor)in0, in1, in2, in3, in4, in5, in6, in7, in8, in9 };
    }

    [Fact]
    public async Task TestModel()
    {
        CompileOptions.DumpFlags = DumpFlags.Rewrite | DumpFlags.PassIR;
        var path = "/Users/homura/Downloads/model_dec.onnx";
        using var file = File.OpenRead(path);
        CompileOptions.InputFormat = Path.GetExtension(file.Name).Trim('.');
        var m = await CompileSession.Compiler.ImportModuleAsync(file);
        m.Entry.InferenceType();
        var mp = ((Function)m.Entry!).VarMap;
        Console.WriteLine("all mp");
        foreach (var (key, value) in mp)
        {
            Console.WriteLine(key.Name);
            Console.WriteLine(value.Select(x => x.ToString().ToArray()));
        }

        Dumpper.DumpIR(m.Entry, "module");
        var pm = CompileSession.CreatePassManager("pm");
        pm.AddWithName<DataflowPass>("pass").Configure(p =>
        {
            p.Add<MatmulToFusion>(mp);
        });
        await pm.RunAsync(m);
        var f = (Function)m.Entry!;
        var types = m.Entry!.ParameterTypes.Select(type => (TensorType)type!).ToArray();
        var inputs = MakeInputs(types);
        var samples = f.Parameters.ToArray().Zip(inputs)
            .ToDictionary(x => x.First, x => (IValue)Value.FromTensor(x.Second));
        TestMatchedCore(f.Body, samples, new FusionBucket(f.VarMap));
    }
}
