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
public class NTTTarget : Target
{
    public const string Kind = "cpu";

    public override string Name => Kind;

    public override IReadOnlyList<IModuleCompiler> ModuleCompilers { get; } = [
        new NTTModuleCompiler(),
    ];

    public override (System.CommandLine.Command Command, Func<InvocationContext, System.CommandLine.Command, ITargetOptions> Parser) RegisterCommandAndParser()
    {
        var cmd = new CpuTargetOptionsCommand(Kind);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var binder = new CpuTargetOptionsBinder(cmd);
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
        // todo config it in the target options.
        var rank = 2;
        var lane = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? 8 : 4;
        pass.Add<Passes.Rules.NTT.PackReduce>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackSwish>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackResizeImage>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackMatMul>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackConv2D>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackUnary>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackBinary>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackTranspose>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackUnsqueeze>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackReshape>(rank, lane);
        pass.Add<Passes.Rules.NTT.PackSlice>(rank, lane);
        pass.Add<Passes.Rules.Neutral.FoldConstCall>();
        pass.Add<Passes.Rules.NTT.FoldPackUnpack>();
        pass.Add<Passes.Rules.NTT.FoldPackConcatUnpack>();
        pass.Add<Passes.Rules.NTT.TransposePackMatMulInputs>();
        pass.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
        pass.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
    }

    public override void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions optionsÍ)
    {
        passManager.Add<NTTTIRSelectionPass>();
    }
}
