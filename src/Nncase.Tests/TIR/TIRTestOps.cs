// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

public sealed class ExtraW : Op
{
    public static readonly ParameterInfo Input = new(typeof(ExtraW), 0, "inputs", TypePatternUtility.IsIntegralScalar());

    public override bool CanFoldConstCall => false;
}

public sealed class ExtraWEvaluator : Evaluator.ITypeInferencer<ExtraW>
{
    public IRType Visit(ITypeInferenceContext context, ExtraW target) => TupleType.Void;
}

public sealed class LoadT : Op
{
    public static readonly ParameterInfo DdrPp = new(typeof(LoadT), 0, "ddr_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());
    public static readonly ParameterInfo GlbPp = new(typeof(LoadT), 1, "glb_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());

    public override bool CanFoldConstCall => false;
}

public sealed class LoadTEvaluator : Evaluator.ITypeInferencer<LoadT>
{
    public IRType Visit(ITypeInferenceContext context, LoadT target) => TupleType.Void;
}

public sealed class StoreT : Op
{
    public static readonly ParameterInfo GlbPp = new(typeof(StoreT), 0, "glb_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());
    public static readonly ParameterInfo DdrPp = new(typeof(StoreT), 1, "ddr_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());

    public override bool CanFoldConstCall => false;
}

public sealed class StoreTEvaluator : Evaluator.ITypeInferencer<StoreT>
{
    public IRType Visit(ITypeInferenceContext context, StoreT target) => TupleType.Void;
}

public sealed class BinaryT : Op
{
    public static readonly ParameterInfo GlbLhsPp = new(typeof(BinaryT), 0, "glb_lhs_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());

    public BinaryT(BinaryOp binaryOp)
    {
        BinaryOp = binaryOp;
    }

    public static readonly ParameterInfo GlbRhsPp = new(typeof(BinaryT), 1, "glb_rhs_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());
    public static readonly ParameterInfo GlbOutPp = new(typeof(BinaryT), 2, "glb_out_pp", TypePatternUtility.IsTensor() | TypePatternUtility.IsAnyType());

    public BinaryOp BinaryOp { get; }

    public override bool CanFoldConstCall => false;
}

public sealed class BinaryTEvaluator : Evaluator.ITypeInferencer<BinaryT>
{
    public IRType Visit(ITypeInferenceContext context, BinaryT target) => TupleType.Void;
}

public sealed class MeshNet : Op
{
    public static readonly ParameterInfo MeshFunc = new(typeof(MeshNet), 0, "meshFunc");

    public static readonly ParameterInfo Input = new(typeof(MeshNet), 1, "input", TypePatternUtility.IsTensor());

    public override bool CanFoldConstCall => false;
}

public sealed class MeshNetEvaluator : Evaluator.ITypeInferencer<MeshNet>, IOpPrinter<MeshNet>
{
    public IRType Visit(ITypeInferenceContext context, MeshNet target) => TupleType.Void;

    public string Visit(IIRPrinterContext context, MeshNet target, bool iLmode)
    {
        return $"I.MeshNet({context.GetArgument(target, MeshNet.MeshFunc).Name}, {context.GetArgument(target, MeshNet.Input)})";
    }
}
