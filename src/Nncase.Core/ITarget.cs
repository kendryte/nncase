﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.Configuration;
using Nncase.CodeGen;
using Nncase.Passes;
using Nncase.Quantization;

namespace Nncase;

/// <summary>
/// Target.
/// </summary>
public interface ITarget
{
    /// <summary>
    /// Gets target kind.
    /// </summary>
    string Kind { get; }

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
    /// Register Target Dependent After Quant Pass.
    /// </summary>
    /// <param name="passManager">Pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterTargetDependentAfterQuantPass(IPassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Target Dependent After Quant Pass.
    /// </summary>
    /// <param name="passManager">Pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options);

    /// <summary>
    /// Create module builder.
    /// </summary>
    /// <param name="moduleKind">Module kind.</param>
    /// <param name="options">compile options.</param>
    /// <returns>Module builder.</returns>
    IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options);
}
