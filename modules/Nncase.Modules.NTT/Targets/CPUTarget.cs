// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;
using Nncase.CodeGen;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Transforms;
using Nncase.Quantization;

namespace Nncase.Targets;

/// <summary>
/// Target for NTT.
/// </summary>
public class CPUTarget : Target
{
    public const string Kind = "cpu";

    private readonly NTTModuleCompiler _nttModuleCompiler = new();

    public CPUTarget()
    {
        ModuleCompilers = [_nttModuleCompiler];
    }

    public override string Name => Kind;

    public override IReadOnlyList<IModuleCompiler> ModuleCompilers { get; }

    public override (System.CommandLine.Command Command, Func<InvocationContext, System.CommandLine.Command, ITargetOptions> Parser) RegisterCommandAndParser()
    {
        var cmd = new NTTTargetOptionsCommand(Kind);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var binder = new NTTTargetOptionsBinder(cmd);
            return binder.GetBoundValue(context);
        }

        return (cmd, ParseTargetCompileOptions);
    }

    public override void RegisterAffineSelectionPass(IPassManager passManager, CompileOptions options)
    {
        passManager.Add<NTTAffineSelectionPass>();
    }

    public override void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options)
    {
        var nr = _nttModuleCompiler.Nr;

        pass.Add<Passes.Rules.NTT.PackMatMulByN>(nr);
    }

    public override void RegisterAutoVectorizeRules(IRulesAddable pass, CompileOptions options)
    {
        // todo config it in the target options.
        var rank = 1;
        var lane = _nttModuleCompiler.Lane;
        var maskVectorStyle = _nttModuleCompiler.MaskVectorStyle;

        pass.Add<Passes.Rules.NTT.VectorizeConv2D>(rank, lane);
        pass.Add<Passes.Rules.NTT.VectorizeMatMul>(rank, lane);
        pass.Add<Passes.Rules.NTT.VectorizeLayerNorm>(rank, lane);

        pass.Add<Passes.Rules.NTT.VectorizeBinaryPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeComparePropagation>(maskVectorStyle);
        pass.Add<Passes.Rules.NTT.VectorizeConcatPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeExpandPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeGatherPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizePadPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeReducePropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeReshapePropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeResizeImagePropagation>();

        // pass.Add<Passes.Rules.NTT.VectorizeScatterND>(rank, lane);
        pass.Add<Passes.Rules.NTT.VectorizeSlicePropagation>();

        // pass.Add<Passes.Rules.NTT.VectorizeSwish>(rank, lane);
        pass.Add<Passes.Rules.NTT.VectorizeTransposePropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeUnaryPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeUnsqueezePropagation>();
        pass.Add<Passes.Rules.NTT.VectorizeWherePropagation>(maskVectorStyle);

        pass.Add<Passes.Rules.NTT.ConcatDevectorizePropagation>();
        pass.Add<Passes.Rules.NTT.BinaryDevectorizeLhsPropagation>();
        pass.Add<Passes.Rules.NTT.BinaryDevectorizeRhsPropagation>();
        pass.Add<Passes.Rules.NTT.VectorizedMatMulDevectorizePropagation>();
        pass.Add<Passes.Rules.NTT.ReshapeDevectorizePropagation>();
        pass.Add<Passes.Rules.NTT.SliceDevectorizePropagation>();
        pass.Add<Passes.Rules.NTT.SwishDevectorizePropagation>();
        pass.Add<Passes.Rules.NTT.TransposeDevectorizePropagation>();
        pass.Add<Passes.Rules.NTT.UnaryDevectorizePropagation>();

        pass.Add<Passes.Rules.Neutral.FoldConstCall>();
        pass.Add<Passes.Rules.NTT.FoldVectorizeDevectorize>();
        pass.Add<Passes.Rules.NTT.FoldVectorizeConcatDevectorize>();
        pass.Add<Passes.Rules.NTT.TransposeVectorizeMatMulInputs>();
        pass.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
        pass.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
    }

    public override void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions optionsÍ)
    {
        passManager.Add<NTTTIRSelectionPass>();
    }
}
