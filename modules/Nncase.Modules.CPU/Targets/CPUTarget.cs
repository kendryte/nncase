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
using Nncase.Passes.Transforms;
using Nncase.Quantization;

namespace Nncase.Targets;

/// <summary>
/// Target for CPU.
/// </summary>
public class CPUTarget : ITarget
{
    public const string Kind = "cpu";

    string ITarget.Kind => Kind;

    public (System.CommandLine.Command Command, Func<InvocationContext, System.CommandLine.Command, ITargetOptions> Parser) RegisterCommandAndParser()
    {
        var cmd = new System.CommandLine.Command(Kind);
        var packingOption = new Option<bool>(
            name: "--packing",
            description: "enable layout optimization.",
            getDefaultValue: () => false);
        cmd.AddOption(packingOption);

        ITargetOptions ParseTargetCompileOptions(InvocationContext context, Command command)
        {
            var packing = context.ParseResult.GetValueForOption(packingOption);
            return new CpuTargetOptions() { Packing = packing };
        }

        return (cmd, ParseTargetCompileOptions);
    }

    /// <inheritdoc/>
    public void ParseTargetDependentOptions(IConfigurationSection configure)
    {
    }

    /// <inheritdoc/>
    public void RegisterTargetInDependentPass(IPassManager passManager, CompileOptions options)
    {
    }

    /// <inheritdoc/>
    public void RegisterTargetDependentPass(IPassManager passManager, CompileOptions options)
    {
    }

    /// <inheritdoc/>
    public Task<Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>> BindQuantMethodCosine(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions)
    {
        var enodeQuantCosineDict = new Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>();
        return Task.FromResult(enodeQuantCosineDict);
    }

    /// <inheritdoc/>
    public Task AdaRoundWeights(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions)
    {
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public void RegisterQuantizePass(IPassManager passManager, CompileOptions options)
    {
    }

    /// <inheritdoc/>
    public void RegisterTargetDependentAfterQuantPass(IPassManager passManager, CompileOptions options)
    {
        if (options.QuantizeOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("RemoveMarker").Configure(p =>
            {
                p.Add<Passes.Rules.Lower.RemoveMarker>();
            });
        }

        if (options.TargetCompileOptions is CpuTargetOptions { Packing: true })
        {
            passManager.AddWithName<EGraphRulesPass>("AutoPacking").Configure(p =>
            {
                // todo config it in the target options.
                var rank = 1;
                var lane = System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? 8 : 4;
                p.Add<Passes.Rules.CPU.PackSoftmax>(rank, lane);
                p.Add<Passes.Rules.CPU.PackSwish>(rank, lane);
                p.Add<Passes.Rules.CPU.PackLayerNorm>(rank, lane);
                p.Add<Passes.Rules.CPU.PackResizeImage>(rank, lane);
                p.Add<Passes.Rules.CPU.PackMatMul>(rank, lane);
                p.Add<Passes.Rules.CPU.PackConv2D>(rank, lane);
                p.Add<Passes.Rules.CPU.PackUnary>(rank, lane);
                p.Add<Passes.Rules.CPU.PackBinary>(rank, lane);
                p.Add<Passes.Rules.CPU.PackTranspose>(rank, lane);
                p.Add<Passes.Rules.CPU.PackUnsqueeze>(rank, lane);
                p.Add<Passes.Rules.CPU.PackReshape>(rank, lane);
                p.Add<Passes.Rules.CPU.PackSlice>(rank, lane);
                p.Add<Passes.Rules.Neutral.FoldConstCall>();
                p.Add<Passes.Rules.CPU.FoldPackUnpack>();
                p.Add<Passes.Rules.CPU.FoldPackConcatUnpack>();
                p.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
            });
        }

        // need refactor tiling.
        // passManager.Add<AutoDistributedPass>();
        passManager.Add<DataflowPass>().Configure(p =>
        {
            p.Add<Passes.Rules.CPU.CPUOutputBoxingFusion>(CPUTarget.Kind);
            p.Add<Passes.Rules.CPU.CPUSingleFusion>(CPUTarget.Kind);
        });
        passManager.Add<DataflowPass>().Configure(p =>
        {
            p.AddAnalysis<Passes.Analysis.IExprUserAnalysisResult>();
            p.Add<Passes.Rules.CPU.DeterminedFusionMergeRule>();
        });
        passManager.AddWithName<EGraphRulesPass>("PartitionConstruct").Configure(p =>
        {
            p.Add<Passes.Rules.CPU.GeneralFusionMergeRule>();
        });
        passManager.AddWithName<EGraphExtractPass>("PartitionExtract").Configure(p =>
        {
            p.AddBaseFuncCostEvaluator<Passes.Rules.CPU.FusionCostEvaluator>();
        });

        // passManager.Add<CPUFunctionPartitionPass>();
        passManager.Add<CPUFusionToModulePass>();

        passManager.AddWithName<DataflowPass>("LowerToAffine").Configure(p =>
        {
            p.Add<Passes.Rules.CPU.Affine.LowerUnary>();
            p.Add<Passes.Rules.CPU.Affine.LowerSwish>();
        });

        // concat/reshape lower
        // tile and lower to tir.
        passManager.Add<AutoTilePass>();

        passManager.Add<CPUFusionToTirPass>();

        // todo add auto fusion merge pass here.
        passManager.Add<PrimFuncPass>().Configure(p =>
        {
            p.Add<Passes.Mutators.UnFoldBlock>();
            p.Add<Passes.Mutators.FlattenSequential>();
            p.Add<Passes.Mutators.TailLoopStripping>();
            p.Add<Passes.Mutators.FoldConstCall>();
        });

        passManager.AddWithName<DDrBufferSchdeulePass>("DDrBufferSchdeule");

        passManager.AddWithName<PrimFuncPass>("InstStage").Configure(p =>
        {
            p.Add<Passes.Mutators.FlattenBuffer>();
            p.Add<Passes.Mutators.FoldConstCall>();
            p.Add<Passes.Mutators.RemoveNop>();
        });
    }

    public void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options)
    {
    }

    /// <inheritdoc/>
    public IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options)
    {
        if (moduleKind == Callable.StackVMModuleKind)
        {
            return new StackVMModuleBuilder();
        }
        else if (moduleKind == "cpu")
        {
            return new CPUModuleBuilder(options);
        }
        else
        {
            throw new NotSupportedException($"{moduleKind} module is not supported.");
        }
    }
}
