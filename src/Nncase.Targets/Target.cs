// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.Configuration;
using Nncase.CodeGen;
using Nncase.Passes;
using Nncase.Quantization;

namespace Nncase.Targets;

public abstract class Target : ITarget
{
    public abstract string Name { get; }

    public virtual IReadOnlyList<IModuleCompiler> ModuleCompilers => Array.Empty<IModuleCompiler>();

    public IModuleCompiler GetModuleCompiler(string moduleKind)
    {
        return ModuleCompilers.Single(m => m.ModuleKind == moduleKind);
    }

    public virtual Task AdaRoundWeights(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions)
    {
        return Task.CompletedTask;
    }

    public virtual Task<Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>> BindQuantMethodCosine(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions)
    {
        return Task.FromResult(new Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>());
    }

    public virtual void ParseTargetDependentOptions(IConfigurationSection configure)
    {
    }

    public abstract (Command Command, Func<InvocationContext, Command, ITargetOptions> Parser) RegisterCommandAndParser();

    public virtual void RegisterQuantizePass(IPassManager passManager, CompileOptions options)
    {
    }

    public virtual void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options)
    {
    }

    public virtual void RegisterTargetDependentPass(IPassManager passManager, CompileOptions options)
    {
    }

    public virtual void RegisterTargetInDependentPass(IPassManager passManager, CompileOptions options)
    {
    }

    public virtual void RegisterAffineSelectionPass(IPassManager passManager, CompileOptions options)
    {
    }

    public virtual void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options)
    {
    }

    public virtual void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions optionsÍ)
    {
    }
}
