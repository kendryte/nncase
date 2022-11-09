﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Nncase.CodeGen;
using Nncase.CodeGen.K210;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Runtime.K210;
using Nncase.Transform;
using Nncase.Transform.Rules.K210;

namespace Nncase.Targets;

/// <summary>
/// Target for K210.
/// </summary>
public class K210Target : ITarget
{
    /// <inheritdoc/>
    public string Kind => "k210";

    /// <inheritdoc/>
    public void ParseTargetDependentOptions(IConfigurationSection configure)
    {
    }

    /// <inheritdoc/>
    public void RegisterTargetDependentPass(PassManager passManager, CompileOptions options)
    {
        if (options.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            passManager.Add(new EGraphPassWithQuantize("lowering_kpu", options.QuantizeOptions!)
            {
                new LowerConv2D(),
            });
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<ENode, List<Tuple<List<DataType>, List<List<QuantParam>>, float>>>> BindQuantMethodCosine(ICalibrationDatasetProvider calibrationDataset, ITarget target, List<ENode> rangeOfs, List<ENode> childrenOfRangeOfs, RunPassOptions runPassOptions)
    {
        return null;
    }

    /// <inheritdoc/>
    public void RegisterQuantizePass(PassManager passManager, CompileOptions options)
    {
    }

    /// <inheritdoc/>
    public void RegisterTargetDependentAfterQuantPass(PassManager passManager, CompileOptions options)
    {
    }

    /// <inheritdoc/>
    public IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options)
    {
        if (moduleKind == KPURTModule.Kind)
        {
            return new KPUModuleBuilder();
        }
        else if (moduleKind == Callable.StackVMModuleKind)
        {
            return new StackVMModuleBuilder();
        }
        else
        {
            throw new NotSupportedException($"{moduleKind} module is not supported.");
        }
    }
}
