﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.Configuration;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Quantization;
using Nncase.Targets;

namespace Nncase;

public enum MemoryAccessArchitecture : byte
{
    /// <summary>
    /// Unified Memory Access.
    /// </summary>
    UMA = 0,

    /// <summary>
    /// Non-Unified Memory Access.
    /// </summary>
    NUMA = 1,
}

public enum NocArchitecture : byte
{
    Mesh = 0,
    CrossBar = 1,
}

public interface ICpuTargetOptions : ITargetOptions
{
    string ModelName { get; set; }

    bool Packing { get; set; }

    bool UnifiedMemoryArch { get; set; }

    MemoryAccessArchitecture MemoryAccessArch { get; set; }

    NocArchitecture NocArch { get; set; }

    HierarchyKind HierarchyKind { get; set; }

    int[][] Hierarchies { get; set; }

    string HierarchyNames { get; set; }

    long[] HierarchySizes { get; set; }

    int[] HierarchyLatencies { get; set; }

    int[] HierarchyBandWidths { get; set; }

    int[] MemoryCapacities { get; set; }

    int[] MemoryBandWidths { get; set; }

    string DistributedScheme { get; set; }

    string CustomOpScheme { get; set; }
}

/// <summary>
/// The targets own compile options.
/// </summary>
public interface ITargetOptions
{
}

/// <summary>
/// Target.
/// </summary>
public interface ITarget
{
    /// <summary>
    /// Gets target kind.
    /// </summary>
    string Name { get; }

    IReadOnlyList<IModuleCompiler> ModuleCompilers { get; }

    IModuleCompiler GetModuleCompiler(string moduleKind);

    /// <summary>
    /// create the current target's command and parser.
    /// </summary>
    /// <returns>command.</returns>
    (System.CommandLine.Command Command, Func<System.CommandLine.Invocation.InvocationContext, System.CommandLine.Command, ITargetOptions> Parser) RegisterCommandAndParser();

    /// <summary>
    /// Bind Quant Method And Quant Cosine With IR.
    /// </summary>
    /// <param name="calibrationDataset">calibration dataset.</param>
    /// <param name="rangeOfs">rangeOf nodes.</param>
    /// <param name="childrenOfRangeOfs">rangeOf nodes children.</param>
    /// <param name="quantizeOptions">options.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    Task<Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>> BindQuantMethodCosine(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions);

    /// <summary>
    /// AdaRound Weights.
    /// </summary>
    /// <param name="calibrationDataset">calibration dataset.</param>
    /// <param name="rangeOfs">rangeOf nodes.</param>
    /// <param name="childrenOfRangeOfs">rangeOf nodes children.</param>
    /// <param name="quantizeOptions">options.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    Task AdaRoundWeights(ICalibrationDatasetProvider calibrationDataset, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, QuantizeOptions quantizeOptions);

    /// <summary>
    /// Parse Target Dependent Options.
    /// </summary>
    void ParseTargetDependentOptions(IConfigurationSection configure);

    /// <summary>
    /// Register Target InDependent Pass.
    /// </summary>
    /// <param name="passManager">pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterTargetInDependentPass(IPassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Target Dependent Pass.
    /// </summary>
    /// <param name="passManager">pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterTargetDependentPass(IPassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Quantize Pass.
    /// </summary>
    /// <param name="passManager">pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterQuantizePass(IPassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Post Quant Pass.
    /// </summary>
    /// <param name="passManager">Pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterPostQuantizePass(IPassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Target Dependent Before CodeGen Pass.
    /// </summary>
    /// <param name="passManager">Pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options);

    void RegisterAffineSelectionPass(IPassManager passManager, CompileOptions options);

    void RegisterAutoPackingRules(IRulesAddable pass, CompileOptions options);

    void RegisterPostAutoPackingPass(IPassManager passManager, CompileOptions options);

    void RegisterTIRSelectionPass(IPassManager passManager, CompileOptions options);
}

public sealed class DefaultTargetCompileOptions : ITargetOptions
{
    public static readonly DefaultTargetCompileOptions Instance = new();

    private DefaultTargetCompileOptions()
    {
    }

    public int[] MemoryCapacities => Array.Empty<int>();

    public int[] MemoryBandWidths => Array.Empty<int>();
}
