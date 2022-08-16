// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.Configuration;
using Nncase.CodeGen;
using Nncase.Quantization;
using Nncase.Transform;

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
    /// Bind Quant Method And Quant Cosine With IR
    /// </summary>
    /// <param name="calibrationDataset">calibration dataset.</param>
    /// <param name="target">target.</param>
    /// <param name="rangeOfs">rangeOf nodes.</param>
    /// <param name="childrenOfRangeOfs">rangeOf nodes children.</param>
    /// <param name="runPassOptions">options.</param>
    Task<Dictionary<ENode, List<Tuple<List<DataType>, List<QuantParam>, float>>>> BindQuantMethodCosine(ICalibrationDatasetProvider calibrationDataset, ITarget target, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, RunPassOptions runPassOptions);

    /// <summary>
    /// Parse Target Dependent Options 
    /// </summary>
    /// <param name="configure"></param>
    void ParseTargetDependentOptions(IConfigurationSection configure);

    /// <summary>
    /// Register Target Dependent Pass
    /// </summary>
    /// <param name="passManager">pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterTargetDependentPass(PassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Quantize Pass
    /// </summary>
    /// <param name="passManager">pass manager.</param>
    /// <param name="options">compile options.</param>
    void RegisterQuantizePass(PassManager passManager, CompileOptions options);

    /// <summary>
    /// Register Target Dependent After Quant Pass
    /// </summary>
    /// <param name="passManager"></param>
    /// <param name="options">compile options.</param>
    void RegisterTargetDependentAfterQuantPass(PassManager passManager, CompileOptions options);

    /// <summary>
    /// Create module builder.
    /// </summary>
    /// <param name="moduleKind">Module kind.</param>
    /// <param name="options">compile options.</param>
    /// <returns>Module builder.</returns>
    IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options);
}
