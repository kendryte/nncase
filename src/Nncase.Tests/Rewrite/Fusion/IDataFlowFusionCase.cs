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

namespace Nncase.Tests.ReWrite.FusionTest;

public interface IDataFlowFusionCase
{
    int FinalFusionCount { get; }

    Expr BuildBody(Var input);
}

public interface IDataFlowFusionCaseTwoStage : IDataFlowFusionCase
{
    int MidFusionCount { get; }
}

internal static class FusionBuilder
{
    private static int _count;

    public static Fusion MakeConv2DFusion(bool mask)
    {
        var fusion_1_input = new Var($"fusion_{_count}_input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, _count, new[] { 3, 3, 1, 1 }).Evaluate().AsTensor();
        var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, _count, new[] { 3 }).Evaluate().AsTensor();
        var fusion_1 = new Fusion(
            $"fusion_{_count}_{mask}",
            Callable.StackVMModuleKind,
            IR.F.NN.Conv2D(
                fusion_1_input,
                weights,
                bias,
                new[] { 1, 1 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1),
            new[] { fusion_1_input });
        _count++;
        return fusion_1;
    }

    public static Fusion MakeBinaryFusion(BinaryOp binaryOp, bool mask)
    {
        var fusion_2_input = new Var[] { new($"fusion_{_count}_input_lhs", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 })), new($"fusion_{_count}_input_rhs", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 })) };
        var fusion_2 = new Fusion($"fusion_{_count}_{mask}", Callable.StackVMModuleKind, IR.F.Math.Binary(binaryOp, fusion_2_input[0], fusion_2_input[1]), fusion_2_input);
        _count++;
        return fusion_2;
    }

    public static Call MakeMultiSingleCall(Expr input, bool[] masks)
    {
        var last_output = input;
        foreach (var mask in masks)
        {
            last_output = new Call(MakeConv2DFusion(mask), last_output);
        }

        return (Call)last_output;
    }
}

/// <summary>
/// cycle type 0:
///  x = fusion1(input)
///  y = fusion2(x)        =>
///  z = fusion3(y)          z = fusion3_2_1(input).
/// </summary>
internal class DataFlowType0FusionCase : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(true), v_1);
        return v_2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }
}

/// <summary>
/// cycle type 0:
///  v0 = fusion1(input)
///  v1 = fusion2(v0)        =>
///  v2 = fusion3(v1)            v2 = fusion3_2_1(input)
///  v3 = fusion4(v2)            v3 = fusion4(v2)
///  v4 = fusion5(v3)            v4 = fusion5(v3).
/// </summary>
internal class DataFlowType0NotFusionCase : IDataFlowFusionCase
{
    public int FinalFusionCount => 3;

    public static Expr BuildBodyCore(Expr input)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(true), v_1);
        var v_3 = new Call(FusionBuilder.MakeConv2DFusion(false), v_2);
        var v_4 = new Call(FusionBuilder.MakeConv2DFusion(true), v_3);
        return v_4;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }
}

/// <summary>
/// cycle type 1:
///             input
///            /    \
///         /         \
///        |      y = fusion2(input)
///         \        /
///          \     /
///     fusion3(x,y).
/// </summary>
internal class DataFlowType1FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
        {
            return new Call(fusion_3, new[] { v_0, input }); // 1,3,224,224
        }

        return new Call(fusion_3, new[] { input, v_0 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

internal sealed class DataFlowType1FusionCaseLeft : DataFlowType1FusionCaseRight
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

/// <summary>
/// cycle type 2:
///             input
///            /    \
///         /         \
///        |      v0 = fusion1(input)
///        |      v1 = fusion2(v0)
///        |      v2 = fusion3(v1)
///         \        /
///          \     /
///     fusion3(input,v2)            =>         fusion?(input).
/// </summary>
internal class DataFlowType2FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(true), v_1);

        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
        {
            return new Call(fusion_3, new[] { v_2, input }); // 1,3,224,224
        }

        return new Call(fusion_3, new[] { input, v_2 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal sealed class DataFlowType2FusionCaseRight : DataFlowType2FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

/// <summary>
/// cycle type 3:
///             input                                      input
///            /    \                                     /    \
///         /         \                                /         \
///        |      v0 = fusion1(input)                 |           |
///        |      v1 = fusion2(v0)                    |      v1 = fusion2_1(input)
///        |      v2 = fusion3(v1)                    |      v2 = fusion3(v1)
///         \        /                                 \        /
///          \     /                                    \     /
///     fusion3(input,v2)            =>              fusion3(input,v2).
/// </summary>
internal class DataFlowType3FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 3;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(false), v_1);

        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
        {
            return new Call(fusion_3, new[] { v_2, input }); // 1,3,224,224
        }

        return new Call(fusion_3, new[] { input, v_2 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal sealed class DataFlowType3FusionCaseRight : DataFlowType3FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

/// <summary>
/// cycle type 3:
///             input                                      input
///            /    \                                     /    \
///         /         \                                /         \
///        |      v0 = fusion1(input)                 |      v1 = fusion2_1(input)
///        |      v1 = fusion2(v0)                    |           |
///        |      v2 = fusion3(v1)                    |      v2 = fusion3(v1)
///        |      v3 = fusion4(v2)                    |           |
///        |      v4 = fusion5(v3)                    |          |
///         \        /                                 \        /
///          \     /                                    \     /
///     fusion6(input,v4)            =>              fusion6_5_4(input,v2).
/// </summary>
internal class DataFlowType4FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 3;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(false), v_1);
        var v_3 = new Call(FusionBuilder.MakeConv2DFusion(true), v_2);
        var v_4 = new Call(FusionBuilder.MakeConv2DFusion(true), v_3);

        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
        {
            return new Call(fusion_3, new[] { v_4, input }); // 1,3,224,224
        }

        return new Call(fusion_3, new[] { input, v_4 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal sealed class DataFlowType4FusionCaseRight : DataFlowType4FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

/// <summary>
/// cycle type 5 = type 2 + fusion:
///             input
///            /    \
///         /         \
///        |      v0 = fusion1(input)
///        |      v1 = fusion2(v0)
///        |      v2 = fusion3(v1)
///         \        /
///          \     /
///     v3 = fusion4(input,v2)            =>       fusion?(input)
///             |
///         fusion5(v3).
/// </summary>
internal class DataFlowType5FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v3 = DataFlowType2FusionCaseLeft.BuildBodyCore(input, left);
        return new Call(FusionBuilder.MakeConv2DFusion(true), new[] { v3 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType5FusionCaseRight : DataFlowType5FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

// cycle type 6 = type 3 + fusion:
//             input                                      input
//            /    \                                     /    \
//         /         \                                /         \
//        |      v0 = fusion1(input)                 |           |
//        |      v1 = fusion2(v0)                    |      v1 = fusion2_1(input)
//        |      v2 = fusion3(v1)                    |      v2 = fusion3(v1)
//         \        /                                 \        /
//          \     /                                    \     /
//     v3 = fusion4(input,v2)            =>            fusion5_4(input,v2)
//             |
//     fusion5(input,v3)
// </summary>
internal class DataFlowType6FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 3;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v3 = DataFlowType3FusionCaseLeft.BuildBodyCore(input, left);
        return new Call(FusionBuilder.MakeConv2DFusion(true), new[] { v3 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType6FusionCaseRight : DataFlowType6FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

// input
//        v0 = fusion0(input)
//        v1 = fusion1(v0)                    v0 = fusion1_0(input)
//            /    \                               /    \
//         /         \                          /         \
//        |      v2 = fusion2(v1)             |            |
//        |      v3 = fusion3(v2)             |       v2 = fusion3_2(v1)
//        |      v4 = fusion4_f(v3)           |       v3 = fusion4_f(v2)
//         \        /                           \        /
//          \     /                              \     /
//     fusion5(input,v4)            =>          fusion9_8(v0,v3)
internal class DataFlowType6_1FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 4;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v1 = new Call(FusionBuilder.MakeConv2DFusion(true), v0);
        var v3 = DataFlowType3FusionCaseLeft.BuildBodyCore(v1, left);

        // return new Call(FusionBuilder.MakeConv2DFusion(true), new[] { v3 }); // 1,3,224,224
        return v3;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType6_1FusionCaseRight : DataFlowType6_1FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

// cycle type 7 : type 5 + 6
//             input
//            /    \
//         /         \
//        |      v0 = fusion0(input)
//        |      v1 = fusion1(v0)
//        |      v2 = fusion2(v1)
//         \        /
//          \     /
//     v3 = fusion3(input,v2)          =>
//     v4 = fusion4(v3)                          v4 = fusion4_3_2_1_0(input)
//            /    \                                     /    \
//         /         \                                /         \
//        |      v5 = fusion5(v4)                    |           |
//        |      v6 = fusion6(v5)                    |      v6 = fusion6_5(v4)
//        |      v7 = fusion7_f(v6)                  |      v7 = fusion7_f(v6)
//         \        /                                 \        /
//          \     /                                    \     /
//     v9 = fusion8(v4,v7)            =>           v10 = fusion9_8(v4,v7)
//             |
//     v10 = fusion9(v9)
// </summary>
internal class DataFlowType7FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 4;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v3 = DataFlowType5FusionCaseLeft.BuildBodyCore(input, left);
        var v9 = DataFlowType6FusionCaseLeft.BuildBodyCore(v3, !left);
        return v9; // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType7FusionCaseRight : DataFlowType7FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

/// <summary>
/// note : fusion2_4 can't fusion
///                   input
///                     |
///                 v0 = fusion0_f(input)
///                     |
///                 v1 = fusion1_f(v0)
///                     |
///                  /    \
///              /           \
///      v2 = fusion2_t(v1)    v3 = fusion3_t(v1)
///           |            \  /         |
///           |            /  \         |
///    v4 = fusion4_t(v2,v3)   v5 = fusion5_t(v2,v3)
///           |                  /
///    v6 = fusion6_f(v4)      /
///           |             /
///    v7 = fusion7_f(v6,v5)
///           |
///    v8 = fusion8_f(v7).
///
/// </summary>
internal class DataFlowType8FusionCase : IDataFlowFusionCase
{
    public int FinalFusionCount => 9;

    public static Expr BuildBodyCore(Expr input)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(false), input);
        var v1 = new Call(FusionBuilder.MakeConv2DFusion(false), v0);

        var v2 = new Call(FusionBuilder.MakeConv2DFusion(true), v1);
        var v3 = new Call(FusionBuilder.MakeConv2DFusion(true), v1);

        var v4 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true), v2, v3);
        var v5 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, true), v2, v3);

        var v6 = new Call(FusionBuilder.MakeConv2DFusion(false), v4);

        var v7 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Min, false), v6, v5);
        var v8 = new Call(FusionBuilder.MakeConv2DFusion(false), v7);
        return v8;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
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
internal class DataFlowType9FusionCase : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v1 = new Call(FusionBuilder.MakeConv2DFusion(true), v0);

        var v2 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true), left ? new[] { v1, v0 } : new[] { v0, v1 });
        var v3 = new Call(FusionBuilder.MakeConv2DFusion(true), v2);

        var v4 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, true), left ? new[] { v3, v2 } : new[] { v2, v3 });
        var v5 = new Call(FusionBuilder.MakeConv2DFusion(true), v4);
        var v6 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Min, true), left ? new[] { v4, v5 } : new[] { v5, v4 });
        return v6;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
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
internal class DataFlowType10FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr inputLhs, Expr inputRhs, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), inputLhs);
        var v1 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true), left ? new[] { v0, inputRhs } : new[] { inputRhs, v0, });
        return v1;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 224, 224 }), true);
    }
}

/// <summary>
/// right version <see cref="DataFlowType10FusionCaseLeft"/>.
/// </summary>
internal sealed class DataFlowType10FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public Expr BuildBody(Var input)
    {
        return DataFlowType10FusionCaseLeft.BuildBodyCore(input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 224, 224 }), false);
    }
}

/// <summary>
/// ShortCutFusionCase, x have one or more user
///                  x
///             /        \
///          /             \
/// v2 = fusion_2_f(x)   v0 = fusion_0(x)   y
///         |               \             /
///         |               \           /
///         |       v1 = fusion_1(v0,y).
///           \         /
///             \     /
///           fusion_3_f(v3,v1).
/// </summary>
internal class DataFlowType10_1FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 3;

    public static Expr BuildBodyCore(Expr inputLhs, Expr inputRhs, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), inputLhs);
        var v1 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true), left ? new[] { v0, inputRhs } : new[] { inputRhs, v0, });
        var v2 = new Call(FusionBuilder.MakeConv2DFusion(false), inputLhs);
        var v4 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, false), left ? new[] { v2, v1 } : new[] { v1, v2 });
        return v4;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 224, 224 }), true);
    }
}

/// <summary>
/// right version <see cref="DataFlowType10FusionCaseLeft"/>.
/// </summary>
internal sealed class DataFlowType10_1FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 3;

    public Expr BuildBody(Var input)
    {
        return DataFlowType10_1FusionCaseLeft.BuildBodyCore(input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 224, 224 }), false);
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
internal class DataFlowType11FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v1 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true), left ? new[] { v0, input } : new[] { input, v0, });
        return v1;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal sealed class DataFlowType11FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public Expr BuildBody(Var input)
    {
        return DataFlowType11FusionCaseLeft.BuildBodyCore(input, false);
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
internal class DataFlowType12FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var y = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 224, 224 });
        var inputRhs = DataFlowType11FusionCaseLeft.BuildBodyCore(y, !left);
        var v1 = DataFlowType10FusionCaseLeft.BuildBodyCore(input, inputRhs, left);
        var v2 = DataFlowType11FusionCaseLeft.BuildBodyCore(v1, !left);
        return v2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType12FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public Expr BuildBody(Var input)
    {
        return DataFlowType12FusionCaseLeft.BuildBodyCore(input, false);
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
internal class DataFlowType13FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v1 = new Call(FusionBuilder.MakeConv2DFusion(true), v0);
        var v2 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, true), left ? new[] { v0, v1 } : new[] { v1, v0 });
        return v2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType13FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public Expr BuildBody(Var input)
    {
        return DataFlowType13FusionCaseLeft.BuildBodyCore(input, false);
    }
}

/// <summary>
///         x
///         |
///       conv2d_f
///       /  \
///       |   conv2d_f
///       |    |
///       |   conv2d_f
///       |    |
///       |   conv2d_t
///        \ /
///       add_f.
/// </summary>
internal class DataFlowType14FusionCaseLeft : IDataFlowFusionCase
{
    public int FinalFusionCount => 4;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(false), input);

        var v1 = new Call(FusionBuilder.MakeConv2DFusion(false), v0);
        var v2 = new Call(FusionBuilder.MakeConv2DFusion(false), v1);
        var v3 = new Call(FusionBuilder.MakeConv2DFusion(true), v2);

        var v4 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, true), left ? new[] { v0, v3 } : new[] { v3, v0 });
        return v4;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType14FusionCaseRight : IDataFlowFusionCase
{
    public int FinalFusionCount => 4;

    public Expr BuildBody(Var input)
    {
        return DataFlowType14FusionCaseLeft.BuildBodyCore(input, false);
    }
}

/// <summary>
///         x
///         |
///       conv2d_t
///       /  \
///       |    |
///       |   conv2d_t
///       |    |
///       |   conv2d_t
///        \ /
///       add_t.
/// </summary>
internal class DataFlowType15FusionCaseLeft : IDataFlowFusionCaseTwoStage
{
    public int FinalFusionCount => 1;

    public int MidFusionCount => 2;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);

        var v1 = new Call(FusionBuilder.MakeConv2DFusion(true), v0);
        var v2 = new Call(FusionBuilder.MakeConv2DFusion(true), v1);

        var v3 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, true), left ? new[] { v0, v2 } : new[] { v2, v0 });
        return v3;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

internal class DataFlowType15FusionCaseRight : IDataFlowFusionCaseTwoStage
{
    public int FinalFusionCount => 1;

    public int MidFusionCount => 2;

    public Expr BuildBody(Var input)
    {
        return DataFlowType15FusionCaseLeft.BuildBodyCore(input, false);
    }
}

/// <summary>
///     x
///    | \
///  f(x, x).
/// </summary>
internal class DataFlowType16FusionCase : IDataFlowFusionCase
{
    public int FinalFusionCount => 1;

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);

        var v3 = new Call(FusionBuilder.MakeBinaryFusion(BinaryOp.Sub, true), new[] { v0, v0 });
        return v3;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}
