// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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
using Nncase.Quantization;

namespace Nncase.Targets;

/// <summary>
/// Target for CPU.
/// </summary>
public class CPUTarget : ITarget
{
    public const string Kind = "cpu";

    string ITarget.Kind => Kind;

    public (System.CommandLine.Command Command, Func<InvocationContext, System.CommandLine.Command, ITargetCompileOptions> Parser) RegisterCommandAndParser()
    {
        return (new System.CommandLine.Command(Kind), (_, _) => DefaultTargetCompileOptions.Instance);
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
        // FIX ME: Disable macos as macho loader is buggy.
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            passManager.AddWithName<DataflowPass>("LowerIR").Configure(p =>
            {
                p.Add<Passes.Rules.CPU.LowerBinary>();
                p.Add<Passes.Rules.CPU.LowerUnary>();
            });
        }
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

        passManager.AddWithName<DataflowPass>("MakeFusion").Configure(p =>
        {
            p.Add<Passes.Rules.CPU.CPUFusion>();
        });

        passManager.Add<CPUFusionToTirPass>(Passes.Tile.TileOptions.Default);

        passManager.Add<PrimFuncPass>().Configure(p =>
        {
            p.Add<Passes.Mutators.UnFoldBlock>();
            p.Add<Passes.Mutators.FlattenSequential>();
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
