// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.PatternMatch;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.TIRTest.PrimFuncTest;

public interface IDataFlowPrimFuncCase
{
    int FinalWrapperCount { get; }

    Expr BuildBody(Var input);
}

internal static class PrimFuncBuilder
{
    public static readonly int[] Dimensions = new[] { 1, 4, 8, 9 };

    private static int _count;

    public static PrimFunctionWrapper MakeLoadStoreFunc(bool mask)
    {
        var allocator = new Allocator();
        var fusion_input = allocator.Allocate($"fusion_{_count}_input", TIR.MemoryLocation.Input);

        var glb = allocator.Allocate($"fusion_{_count}_glb", TIR.MemoryLocation.L2Data);

        var fusion_output = allocator.Allocate($"fusion_{_count}_output", TIR.MemoryLocation.Output);

        var fusion_1 = TIR.T.PrimFunc($"fusion_{_count}_{mask}", Callable.StackVMModuleKind, fusion_input, fusion_output).Body(
          new Call(new TIRTest.LoadT(), fusion_input, glb),
          new Call(new TIRTest.LoadT(), glb, fusion_output)).Build();

        _count++;
        return new PrimFunctionWrapper($"fusion_{_count}_{mask}_w", fusion_1, 1);
    }

    public static PrimFunctionWrapper MakeBinaryFunc(BinaryOp binaryOp, bool mask)
    {
        var allocator = new Allocator();
        var fusion_input_lhs = allocator.Allocate($"fusion_{_count}_input_lhs", TIR.MemoryLocation.Input);
        var fusion_input_rhs = allocator.Allocate($"fusion_{_count}_input_rhs", TIR.MemoryLocation.Input);
        var glb_lhs = allocator.Allocate($"fusion_{_count}_glb_lhs", TIR.MemoryLocation.L2Data);
        var glb_rhs = allocator.Allocate($"fusion_{_count}_glb_rhs", TIR.MemoryLocation.L2Data);
        var glb_output = allocator.Allocate($"fusion_{_count}_glb_output", TIR.MemoryLocation.L2Data);
        var fusion_output = allocator.Allocate($"fusion_{_count}_output", TIR.MemoryLocation.Output);

        var fusion = TIR.T.PrimFunc($"fusion_{_count}_{mask}", Callable.StackVMModuleKind, fusion_input_lhs, fusion_input_rhs, fusion_output).Body(
          new Call(new TIRTest.LoadT(), fusion_input_lhs, glb_lhs),
          new Call(new TIRTest.LoadT(), fusion_input_rhs, glb_rhs),
          new Call(new TIRTest.BinaryT(binaryOp), glb_lhs, glb_rhs, glb_output),
          new Call(new TIRTest.StoreT(), glb_output, fusion_output)).Build();

        var wrapper = new PrimFunctionWrapper($"fusion_{_count}_{mask}_w", fusion, 2);
        _count++;
        return wrapper;
    }

    public static PrimFunctionWrapper MakeMultiInputFunc(int length, bool mask)
    {
        var allocator = new Allocator();
        var fusion_inputs = new List<TIR.Buffer>();
        for (int i = 0; i < length; i++)
        {
            var fusion_input_i = allocator.Allocate($"fusion_{_count}_input_{i}", TIR.MemoryLocation.Input);
            fusion_inputs.Add(fusion_input_i);
        }

        var glb1 = allocator.Allocate($"fusion_{_count}_glb1", TIR.MemoryLocation.L2Data);
        var glb2 = allocator.Allocate($"fusion_{_count}_glb2", TIR.MemoryLocation.L2Data);
        var fusion_output = allocator.Allocate($"fusion_{_count}_output", TIR.MemoryLocation.Output);

        var fusion = TIR.T.PrimFunc($"multi_fusion_{_count}_{mask}", Callable.StackVMModuleKind, fusion_inputs.Concat(new[] { fusion_output }).ToArray());

        fusion.Body(
          new Call(new LoadT(), fusion_inputs[0], glb1));
        var last = glb1;
        foreach (var (b, i) in GetBinaryOp(length - 1).Select((b, i) => (b, i)))
        {
            fusion.Body(
              new Call(new LoadT(), fusion_inputs[i + 1], glb2),
              new Call(new BinaryT(b), glb1, glb2, glb1));
        }

        fusion.Body(
          new Call(new TIRTest.StoreT(), glb1, fusion_output));

        var wrapper = new PrimFunctionWrapper($"multi_fusion_{_count}_{mask}_w", fusion.Build(), length);
        _count++;
        return wrapper;
    }

    public static Call MakeMultiSingleCall(Expr input, bool[] masks)
    {
        var last_output = input;
        foreach (var mask in masks)
        {
            last_output = new Call(MakeLoadStoreFunc(mask), last_output);
        }

        return (Call)last_output;
    }

    private static IEnumerable<BinaryOp> GetBinaryOp(int length)
    {
        var ops = new[] { BinaryOp.Add, BinaryOp.Mul, BinaryOp.Sub, BinaryOp.Mul };
        for (int i = 0; i < length; i++)
        {
            yield return ops[i % ops.Length];
        }
    }

    private sealed class Allocator
    {
        private readonly Dictionary<TIR.MemoryLocation, ulong> _usage = new() {
          { TIR.MemoryLocation.Input, 0 },
          { TIR.MemoryLocation.Output, 0 },
          { TIR.MemoryLocation.L2Data, 0 },
        };

        public TIR.Buffer Allocate(string name, TIR.MemoryLocation location)
        {
            var dims = Dimensions.Select(d => (Expr)d).ToArray();
            var strides = TensorUtilities.GetStrides(Dimensions).Select(s => (Expr)s).ToArray();
            var size = TensorUtilities.GetSize(Dimensions, TensorUtilities.GetStrides(Dimensions), DataTypes.Float32.SizeInBytes);

            var buffer = new TIR.Buffer(name, DataTypes.Float32, new TIR.MemSpan(Tensor.FromPointer<float>(_usage[location]), size, location), dims, strides);
            _usage[location] += (ulong)size;
            return buffer;
        }
    }
}

/// <summary>
/// cycle type 0:
///  x = fusion1(input)
///  y = fusion2(x)        =>
///  z = fusion3(y)          z = fusion3_2_1(input).
/// </summary>
internal sealed class DataFlowType0FusionCase : IDataFlowPrimFuncCase
{
    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input)
    {
        var v_0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var v_1 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v_0);
        var v_2 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v_1);
        return v_2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }
}

/// <summary>
///   x
///   |
/// fusion1 y   =>   x     y
///  \     /          \   /
///   fusion2        fusion1_2.
/// </summary>
internal sealed class DataFlowType1FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType1FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var y = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, PrimFuncBuilder.Dimensions);
        var v_1 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Add, true), left ? new[] { v_0, y } : new[] { y, v_0 });
        return v_1;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}

/// <summary>
///   x
///   |
/// fusion1 y   =>   x     y
///  \     /          \   /
///   fusion2        fusion1_2_3
///     |
///   fusion3.
/// </summary>
internal sealed class DataFlowType2FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType2FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_2 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), DataFlowType1FusionCase.BuildBodyCore(input, left));
        return v_2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}

/// <summary>
/// current not support
///        y
///        |
///     fusion1               y
///   /       \              |
///  \        /    =>       |
///   fusion2           fusion2_1.
///
/// </summary>
internal sealed class DataFlowType3FusionCase : IDataFlowPrimFuncCase
{
    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input)
    {
        var v_0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var v_1 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Add, true), new[] { v_0, v_0 });
        return v_1;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }
}

/// <summary>
///                       in2
///                        |
///                      func_1           in4
///         in1         /      \            |
///          |        /         \          |
///        func0    /    in3     |     func_3
///          |     |      |      |        |
///  func_4(            func_2            )
///
///                                in4
///         in1                     |
///          |                     |
///        func0  in2    in3     func_3
///          |     |      |       |
///  func_5_1(          func_2      ).
///
///
/// </summary>
internal sealed class DataFlowType4FusionCase : IDataFlowPrimFuncCase
{
    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr in0)
    {
        var v_0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), in0);

        var in1 = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, PrimFuncBuilder.Dimensions);
        var v_1 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), in1);

        var in2 = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, PrimFuncBuilder.Dimensions);
        var v_2 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), in2);

        var in3 = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, PrimFuncBuilder.Dimensions);
        var v_3 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), in3);

        var v_4 = new Call(PrimFuncBuilder.MakeMultiInputFunc(5, true), v_0, v_1, v_2, v_1, v_3);
        return v_4;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }

    private bool CheckOnFinish(TestVisitor v)
    {
        var wrapper = v.ExprMemo.OfType<PrimFunctionWrapper>().Where(p => p.Name.StartsWith("multi_fusion_")).First();
        return wrapper.ParametersCount == 3;
    }
}

/// <summary>
///       x
///       |
///    func_0
///       |   \           x
///    func_1  |    ->    |
///       |   /         func_3
///    func_2
///       |
///    func_3.
/// </summary>
internal sealed class DataFlowType5FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType5FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var v_1 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v_0);
        var v_2 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Add, true), left ? new[] { v_1, v_0 } : new[] { v_0, v_1 });
        var v_3 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v_2);
        return v_3;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }

    private bool CheckOnFinish(TestVisitor v)
    {
        var wrapper = v.ExprMemo.OfType<PrimFunctionWrapper>().First();
        return wrapper.ParametersCount == 1;
    }
}

/// <summary>
///   input
///     |
/// v0 = fusion0(input)
///     |              \
/// v1 = fusion1(v0)   |
///         \          |
///           \       /
///        v2 = fusion2(v1,v0)
///             |            \
///        v3 = fusion3(v2)   |
///                   \       |
///                      \    |
///           v4 = fusion4(v3,v2)
///                |              \
///             v5 = fusion5(v4)   |
///                  |            /
///             v6 = fusion6(v5,v4).
/// </summary>
internal sealed class DataFlowType6FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType6FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var v1 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v0);

        var v2 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Add, true), left ? new[] { v1, v0 } : new[] { v0, v1 });
        var v3 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v2);

        var v4 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Sub, true), left ? new[] { v3, v2 } : new[] { v2, v3 });
        var v5 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v4);
        var v6 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Min, true), left ? new[] { v4, v5 } : new[] { v5, v4 });
        return v6;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}

/// <summary>
/// ShortCutFusionCase
///           x
///          /
/// v0 = fusion_0(x)      y
///          \           /
///            \       /
///     v1 = fusion_1(v0,y).
/// </summary>
internal sealed class DataFlowType7FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType7FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr inputLhs, bool left)
    {
        var inputRhs = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, PrimFuncBuilder.Dimensions);
        var v0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), inputLhs);
        var v1 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Add, true), left ? new[] { v0, inputRhs } : new[] { inputRhs, v0, });
        return v1;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}

/// <summary>
/// ShortCutFusionCase
///                x
///              /    \
/// v0 = fusion_0(x)    \
///          \           |
///            \         /
///     v1 = fusion_1(v0,x).
/// </summary>
internal sealed class DataFlowType8FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType8FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var v1 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Add, true), left ? new[] { v0, input } : new[] { input, v0, });
        return v1;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}

/// <summary>
/// ShortCutFusionCase
///                            y
///           x              |   \
///          /         fusion_0() \
///          /               |     |
/// v0 = fusion_0(x)   fusion_0( ,y)
///          \           /
///            \       /
///     v1 = fusion_1(v0,y)
///               |        \
///     v2 = fusion_2(v1)   |
///               |        /
///     v3 = fusion_3(v2,v1).
/// </summary>
internal sealed class DataFlowType9FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType9FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var y = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 7, PrimFuncBuilder.Dimensions);
        _ = DataFlowType8FusionCase.BuildBodyCore(y, !left);
        var v1 = DataFlowType7FusionCase.BuildBodyCore(input, left);
        var v2 = DataFlowType8FusionCase.BuildBodyCore(v1, !left);
        return v2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}

/// <summary>
///       x
///       |
///    v0 = fusion0(x)
///      |           |
///      |     v1 =fusion1(v0)
///    v2 = fusion2(v0,v1).
///
/// </summary>
internal sealed class DataFlowType10FusionCase : IDataFlowPrimFuncCase
{
    private readonly bool _left;

    public DataFlowType10FusionCase(bool left)
    {
        _left = left;
    }

    public int FinalWrapperCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), input);
        var v1 = new Call(PrimFuncBuilder.MakeLoadStoreFunc(true), v0);
        var v2 = new Call(PrimFuncBuilder.MakeBinaryFunc(BinaryOp.Sub, true), left ? new[] { v0, v1 } : new[] { v1, v0 });
        return v2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, _left);
    }
}
