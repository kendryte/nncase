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
using Nncase.CodeGen.CPU;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Transforms;
using Nncase.Quantization;

namespace Nncase.Targets;

/// <summary>
/// Target for CPU.
/// </summary>
public class CPUTarget : Target
{
    public const string Kind = "cpu";

    public override string Name => Kind;

    public override IReadOnlyList<IModuleCompiler> ModuleCompilers { get; } = [
        new CPUModuleCompiler(),
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
        passManager.Add<CPUAffineSelectionPass>();
    }

    public override void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options)
    {
        // todo config it in the target options.
        var rank = 1;
        var lane = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? 32 : 16;
        pass.Add<Passes.Rules.CPU.PackReduce>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackSwish>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackResizeImage>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackMatMul>(2, lane);
        pass.Add<Passes.Rules.CPU.PackConv2D>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackUnary>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackBinary>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackTranspose>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackUnsqueeze>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackReshape>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackSlice>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackGather>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackCompare>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackConcat>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackExpand>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackWhere>(rank, lane);
        pass.Add<Passes.Rules.CPU.PackScatterND>(rank, lane);
        pass.Add<Passes.Rules.Neutral.FoldConstCall>();
        pass.Add<Passes.Rules.CPU.FoldPackUnpack>();
        pass.Add<Passes.Rules.CPU.FoldPackConcatUnpack>();
        pass.Add<Passes.Rules.CPU.TransposePackMatMulInputs>();
        pass.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
        pass.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
    }

    public override void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions optionsÍ)
    {
        passManager.Add<CPUTIRSelectionPass>();
    }
}
