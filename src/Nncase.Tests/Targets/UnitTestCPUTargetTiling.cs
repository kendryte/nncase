// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Runtime.Interop;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;
using static Nncase.IR.F.Tensors;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Tests.Targets;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCPUTargetTiling : TestClassBase
{
    public UnitTestCPUTargetTiling()
    {
        DefaultTargetName = CPUTarget.Kind;
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    [Theory]

    // [ClassData(typeof(TilingCaseMHA))]
    // [ClassData(typeof(TilingCaseBinaryMul))]
    // [ClassData(typeof(TilingCaseUnary))]
    // [ClassData(typeof(TilingCaseMatmul))]
    // [ClassData(typeof(TilingCaseLayerNorm))]
    // [ClassData(typeof(TilingCaseGather))]
    // [ClassData(typeof(TilingCaseSoftmax))]
    // [ClassData(typeof(TilingCaseSlice))]
    // [ClassData(typeof(TilingCaseConcat))]
    // [ClassData(typeof(TilingCaseTranspose))]
    // [ClassData(typeof(TilingCaseReshape1))]
    // [ClassData(typeof(TilingCaseReshape2))]
    // [ClassData(typeof(TilingCaseMatmulUnary))]
    // [ClassData(typeof(TilingCaseConv2D))]
    // [ClassData(typeof(TilingCaseReduceArg))]
    // [ClassData(typeof(TilingCaseReduceArg2))]
    // [ClassData(typeof(TilingCaseInstanceNorm))]
    // [ClassData(typeof(TilingCaseEncoderTail))]
    [ClassData(typeof(TilingCaseResize))]
    public async Task TestCpuFunction(Function main, Tensor[] inputs)
    {
        var module = new IR.IRModule(main);
        using (new Diagnostics.DumpScope(main.Name, CompileOptions.DumpFlags))
        {
#if DEBUG
            for (var i = 0; i < inputs.Length - 1; i++)
            {
                using (var fs = Diagnostics.DumpScope.Current.OpenFile($"input_{i}.bin"))
                {
                    fs.Write(inputs[i].BytesBuffer);
                }
            }

            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"output_0.bin"))
            {
                fs.Write(inputs[^1].BytesBuffer);
            }
#endif
            await Compile(module);
            var output = Testing.RunKModel(File.ReadAllBytes(Path.Join(Diagnostics.DumpScope.Current.Directory, "test.kmodel")), Diagnostics.DumpScope.Current.Directory, inputs);
#if DEBUG
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"actual_0.bin"))
            {
                fs.Write(output.AsTensor().BytesBuffer);
            }
#endif
            var cos = Tests.Comparator.CosSimilarity(output, Value.FromTensor(inputs[^1]))[0];
            Assert.True(cos > 0.999);
        }
    }

    private async Task Compile(IRModule module)
    {
        var compiler = CompileSession.Compiler;
        compiler.ImportIRModule(module);
        await compiler.CompileAsync();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("test.kmodel"))
        {
            compiler.Gencode(fs);
        }
    }
}

internal sealed class TilingCaseSlice : TheoryData<Function, Tensor[]>
{
    public TilingCaseSlice()
    {
        var shape = new[] { 1, 1, 64, 128 };
        var in_a = new Var("in_a", new TensorType(DataTypes.Float32, shape));

        Fusion fusion;
        Var fin_a;
        {
            fin_a = new Var("fin_a", new TensorType(DataTypes.Float32, shape));
            var v1 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Slice()), fin_a, new[] { 64 }, new[] { 128 }, new[] { 3 }, new[] { 1 });
            fusion = new Fusion("cpu", v1, fin_a);
        }

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin_a, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        var main = new Function("slice", new Call(fusion, in_a), new[] { in_a });
        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseConcat : TheoryData<Function, Tensor[]>
{
    public TilingCaseConcat()
    {
        var in_a_shape = new[] { 1, 64, 384, 64 };
        var in_b_shape = new[] { 1, 64, 384, 64 };
        var in_a = new Var("in_a", new TensorType(DataTypes.Float32, in_a_shape));
        var in_b = new Var("in_b", new TensorType(DataTypes.Float32, in_b_shape));

        Fusion fusion;
        Var fin_a, fin_b;
        {
            fin_a = new Var("fin_a", new TensorType(DataTypes.Float32, in_a_shape));
            fin_b = new Var("fin_b", new TensorType(DataTypes.Float32, in_b_shape));
            var v1 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Concat(3)), new IR.Tuple(fin_a, fin_b));
            fusion = new Fusion("cpu", v1, fin_a, fin_b);
        }

        var input_a_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, in_a_shape).Evaluate().AsTensor();
        var input_b_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, in_b_shape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin_a, Value.FromTensor(input_a_tensor) },
            { fin_b, Value.FromTensor(input_b_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        var main = new Function("concat", new Call(fusion, in_a, in_b), new[] { in_a, in_b });
        Add(main, new[] { input_a_tensor, input_b_tensor, output });
    }
}

internal sealed class TilingCaseMatmulLayerNorm : TheoryData<Function, Tensor[]>
{
    public TilingCaseMatmulLayerNorm()
    {
        var hid_in = new Var("hidden_in", new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 }));

        Fusion fusion;
        {
            var scale = IR.F.Tensors.ConstantOfShape(new[] { 8192 }, 1.0f).Evaluate().AsTensor();
            var bias = IR.F.Tensors.ConstantOfShape(new[] { 8192 }, 0.0f).Evaluate().AsTensor();
            var weights = IR.F.Random.Normal(DataTypes.Float32, new[] { 1, 64, 8192, 128 }).Evaluate().AsTensor();
            _ = IR.F.Random.Normal(DataTypes.Float32, new[] { 384, 128 }).Evaluate().AsTensor();

            var fin = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 }));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.NN.LayerNorm(2, 1e-6f, false)), fin, scale, bias);
            var v1 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Unsqueeze()), v0, new[] { 0 });
            var v2 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.MatMul()), v1, weights);
            var v3 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.Unary(UnaryOp.Exp)), v2);

            fusion = new Fusion("cpu", v3, fin);
        }

        var main = new Function("matmul_layernorm", new Call(fusion, hid_in), new[] { hid_in });
        Add(main, Array.Empty<Tensor>());
    }
}

internal sealed class TilingCaseMHA : TheoryData<Function, Tensor[]>
{
    public TilingCaseMHA()
    {
        var hid_in = new Var("hidden_in", new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 }));
        var pos_ids = new Var("position_ids", new TensorType(DataTypes.Int64, new[] { 1, 384 }));

        Fusion fusion;
        {
            var scale = IR.F.Tensors.ConstantOfShape(new[] { 8192 }, 1.0f).Evaluate().AsTensor();
            var bias = IR.F.Tensors.ConstantOfShape(new[] { 8192 }, 0.0f).Evaluate().AsTensor();
            var weights = IR.F.Random.Normal(DataTypes.Float32, new[] { 1, 64, 8192, 128 }).Evaluate().AsTensor();
            var gdata = IR.F.Random.Normal(DataTypes.Float32, new[] { 384, 128 }).Evaluate().AsTensor();

            var fin = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 }));
            var fin2 = new Var("input2", new TensorType(DataTypes.Int64, new[] { 1, 384 }));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.NN.LayerNorm(2, 1e-6f, false)), fin, scale, bias);
            var v1 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Unsqueeze()), v0, new[] { 0 });
            var v2 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.MatMul()), v1, weights);
            var v3 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Slice()), v2, new[] { 64 }, new[] { 128 }, new[] { 3 }, new[] { 1 });
            var v4 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.Unary(UnaryOp.Neg)), v3);
            var v5 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Slice()), v2, new[] { 0 }, new[] { 64 }, new[] { 3 }, new[] { 1 });
            var v6 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Concat(3)), new IR.Tuple(v4, v5));

            var v7 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Gather(0)), gdata, fin2);
            var v8 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Unsqueeze()), v7, new[] { 0 });

            var v9 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.Binary(BinaryOp.Mul)), v2, v8);
            var v10 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.Binary(BinaryOp.Mul)), v6, v8);
            var v11 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.Binary(BinaryOp.Add)), v9, v10);

            fusion = new Fusion("cpu", v11, fin, fin2);
        }

        var main = new Function("mha_qk", new Call(fusion, hid_in, pos_ids), new[] { hid_in, pos_ids });
        Add(main, Array.Empty<Tensor>());
    }
}

internal sealed class TilingCaseBinaryMul : TheoryData<Function, Tensor[]>
{
    public TilingCaseBinaryMul()
    {
        var lhsShape = new[] { 1, 64, 384, 128 };
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, lhsShape));
        var rhsShape = new[] { 1, 1, 384, 128 };
        var rhs = new Var("lhs", new TensorType(DataTypes.Float32, rhsShape));
        var main = new Function("binary_mul", lhs * rhs, new[] { lhs, rhs });

        var lhs_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, lhsShape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(lhs_tensor.BytesBuffer);
        }

        var rhs_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_1.bin"))
        {
            fs.Write(rhs_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
        {
            { lhs, Value.FromTensor(lhs_tensor) },
            { rhs, Value.FromTensor(rhs_tensor) },
        };
        var output = main.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { lhs_tensor, rhs_tensor, output });
    }
}

internal sealed class TilingCaseUnary : TheoryData<Function, Tensor[]>
{
    public TilingCaseUnary()
    {
        var shape = new[] { 1, 384, 2048 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var main = new Function("unary", IR.F.Math.Unary(UnaryOp.Asin, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(input_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
        {
            { input, Value.FromTensor(input_tensor) },
        };
        var output = main.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseMatmul : TheoryData<Function, Tensor[]>
{
    public TilingCaseMatmul()
    {
        var lhsShape = new[] { 1, 64, 384, 8192 };
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, lhsShape));
        var rhsShape = new[] { 1, 64, 8192, 128 };
        var rhs = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, rhsShape).Evaluate().AsTensor().Cast<float>();

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, lhsShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.MatMul()), fin, rhs);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("matmul", new Call(fusion, lhs), new[] { lhs });

        var lhs_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, lhsShape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(lhs_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(lhs_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { lhs_tensor, output });
    }
}

internal sealed class TilingCaseMatmulUnary : TheoryData<Function, Tensor[]>
{
    public TilingCaseMatmulUnary()
    {
        var lhsShape = new[] { 1, 64, 384, 8192 };
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, lhsShape));
        var rhsShape = new[] { 1, 64, 8192, 128 };
        var rhs = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, rhsShape).Evaluate().AsTensor().Cast<float>();
        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, lhsShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.MatMul()), fin, rhs);
            var v1 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.Unary(UnaryOp.Neg)), v0);
            fusion = new Fusion("cpu", v1, fin);
        }

        var main = new Function("matmul_unary", new Call(fusion, lhs), new[] { lhs });

        var lhs_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, lhsShape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(lhs_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(lhs_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { lhs_tensor, output });
    }
}

internal sealed class TilingCaseLayerNorm : TheoryData<Function, Tensor[]>
{
    public TilingCaseLayerNorm()
    {
        var shape = new[] { 1, 384, 8192 };
        int axis = 2;
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var scale = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new[] { shape[2] }).Evaluate().AsTensor();
        var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, new[] { shape[2] }).Evaluate().AsTensor();

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, shape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.NN.LayerNorm(axis, 1e-5f, false)), fin, scale, bias);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("layernorm", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(input_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseInstanceNorm : TheoryData<Function, Tensor[]>
{
    public TilingCaseInstanceNorm()
    {
        var shape = new[] { 1, 32, 65536 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var scale = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new[] { shape[1] }).Evaluate().AsTensor();
        var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, new[] { shape[1] }).Evaluate().AsTensor();

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, shape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.NN.InstanceNormalization()), fin, scale, bias, 1e-5);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("instance_norm", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(input_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseGather : TheoryData<Function, Tensor[]>
{
    public TilingCaseGather()
    {
        var inputShape = new[] { 384, 128 };
        var axisShape = new[] { 1, 384 };
        int axis = 0;
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        var indices = IR.F.Random.Uniform(DataTypes.Int64, 384, 0, 2, axisShape).Evaluate().AsTensor();

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Gather(axis)), fin, indices);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("gather", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseSoftmax : TheoryData<Function, Tensor[]>
{
    public TilingCaseSoftmax()
    {
        var inputShape = new[] { 1, 64, 384, 384 };
        int axis = 3;
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.NN.Softmax()), fin, axis);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("softmax", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseTranspose : TheoryData<Function, Tensor[]>
{
    public TilingCaseTranspose()
    {
        var shape = new[] { 1, 1, 64, 128 };
        var in_a = new Var("in_a", new TensorType(DataTypes.Float32, shape));
        var perm = new[] { 0, 2, 3, 1 };

        Fusion fusion;
        Var fin_a;
        {
            fin_a = new Var("fin_a", new TensorType(DataTypes.Float32, shape));
            var v1 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Transpose()), fin_a, perm);
            fusion = new Fusion("cpu", v1, fin_a);
        }

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin_a, Value.FromTensor(input_tensor) },
        };
        _ = fusion.Body.Evaluate(feedDict).AsTensor();
        _ = new Function("transpose", new Call(fusion, in_a), new[] { in_a });
    }
}

internal sealed class TilingCaseReshape1 : TheoryData<Function, Tensor[]>
{
    public TilingCaseReshape1()
    {
        var inputShape = new[] { 1, 384, 128 };
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        var newShape = new[] { 1, 1, 384, 128 };

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Reshape()), fin, newShape);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("reshape1", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseReshape2 : TheoryData<Function, Tensor[]>
{
    public TilingCaseReshape2()
    {
        var inputShape = new[] { 1, 384, 64, 128 };
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        var newShape = new[] { 1, 384, 8192 };

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Tensors.Reshape()), fin, newShape);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("reshape2", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseReduceArg : TheoryData<Function, Tensor[]>
{
    public TilingCaseReduceArg()
    {
        var inputShape = new[] { 1, 77 };
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64)), fin, 1, false, false);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("argmax", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseReduceArg2 : TheoryData<Function, Tensor[]>
{
    public TilingCaseReduceArg2()
    {
        var inputShape = new[] { 32, 64, 128, 512 };
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Math.ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64)), fin, 2, true, false);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("argmax2", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseEncoderTail : TheoryData<Function, Tensor[]>
{
    public TilingCaseEncoderTail()
    {
        var v_16 = new Var("v16", new TensorType(DataTypes.Int32, new[] { 1, 77 }));
        var v_17 = new Var("v17", new TensorType(DataTypes.Float32, new[] { 1, 77, 768 }));
        var v0 = IR.F.NN.LayerNorm(2, 1E-05f, v_17, IR.F.Tensors.ConstantOfShape(new[] { 768 }, 1.0f), IR.F.Tensors.ConstantOfShape(new[] { 768 }, 0.0f), true); // f32[1,77,768]
        var v1 = IR.F.Tensors.Reshape(v0, new[] { 77, 768 }); // f32[77,768]
        var v2 = IR.F.Tensors.ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, v_16, -1, false, false); // i64[1]
        var v3 = IR.F.Math.Binary(BinaryOp.Add, v2, 1L); // i64[1]
        var v4 = IR.F.Tensors.Gather(v1, 0, v3); // f32[1,768]
        var main = new Function("encoderTail", v4, new[] { v_16, v_17 });

        var input_16 = IR.F.Tensors.Cast(IR.F.Random.Uniform(DataTypes.Float32, 60, 1, 2, new[] { 1, 77 }), DataTypes.Int32).Evaluate().AsTensor();
        var input_17 = IR.F.Random.Normal(DataTypes.Float32, 2, 4, 3, new[] { 1, 77, 768 }).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { v_16, Value.FromTensor(input_16) },
            { v_17, Value.FromTensor(input_17) },
        };
        var output = main.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_16, input_17, output });
    }
}

internal sealed class TilingCaseConv2D : TheoryData<Function, Tensor[]>
{
    public TilingCaseConv2D()
    {
        var inputShape = new[] { 8, 4, 64, 64 };
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        var weightsShape = new[] { 1, 4, 3, 3 };
        var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, weightsShape).Evaluate().AsTensor().Cast<float>();
        var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 2, 2, weightsShape.Take(1).ToArray()).Evaluate().AsTensor().Cast<float>();
        var stride = new int[] { 1, 1 };
        var padding = new int[,] { { 0, 0 }, { 0, 0 } };
        var dilation = new int[] { 1, 1 };
        var groups = 1;
        var fusedClamp = new float[] { float.NegativeInfinity, float.PositiveInfinity };

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, inputShape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.NN.Conv2D(PadMode.Constant)), fin, weights, bias, stride, padding, dilation, groups, fusedClamp);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("conv2d", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate().AsTensor();
        var feedDict = new Dictionary<Var, IValue>
        {
            { fin, Value.FromTensor(input_tensor) },
        };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}

internal sealed class TilingCaseResize : TheoryData<Function, Tensor[]>
{
    public TilingCaseResize()
    {
        var shape = new[] { 1, 512, 64, 64 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));

        Fusion fusion;
        Var fin;
        {
            fin = new Var("input", new TensorType(DataTypes.Float32, shape));
            var v0 = new Call(new IR.CPU.CPUKernelOp(new IR.Imaging.ResizeImage(ImageResizeMode.NearestNeighbor, ImageResizeTransformationMode.Asymmetric, ImageResizeNearestMode.Floor, false)), fin, None.Default, new[] { 1, 512, 128, 128 }, None.Default, None.Default, None.Default);
            fusion = new Fusion("cpu", v0, fin);
        }

        var main = new Function("nearest_resize", new Call(fusion, input), new[] { input });

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        using (var fs = Diagnostics.DumpScope.Current.OpenFile("input_0.bin"))
        {
            fs.Write(input_tensor.BytesBuffer);
        }

        var feedDict = new Dictionary<Var, IValue>
            {
                { fin, Value.FromTensor(input_tensor) },
            };
        var output = fusion.Body.Evaluate(feedDict).AsTensor();

        Add(main, new[] { input_tensor, output });
    }
}
