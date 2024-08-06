﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine.Invocation;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;
using Nncase.CodeGen;
using Nncase.CodeGen.Ncnn;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
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
        passManager.AddWithName<DataflowPass>("LowerNcnnIR").Configure(p =>
        {
            p.Add<Passes.Rules.Lower.RemoveMarker>();

            // Fold reduce
            p.Add<Passes.Rules.Ncnn.LowerReductionSumSquare>();
            p.Add<Passes.Rules.Ncnn.LowerReductionL1>();
            p.Add<Passes.Rules.Ncnn.LowerReductionL2>();
            p.Add<Passes.Rules.Ncnn.LowerReductionLogSum>();
            p.Add<Passes.Rules.Ncnn.LowerReductionLogSumExp>();

            // single op
            p.Add<Passes.Rules.Ncnn.LowerBatchNorm>();
            p.Add<Passes.Rules.Ncnn.LowerSoftmax>();
            p.Add<Passes.Rules.Ncnn.LowerUnary>();
            p.Add<Passes.Rules.Ncnn.LowerBinary>();

            p.Add<Passes.Rules.Ncnn.LowerCelu>();
            p.Add<Passes.Rules.Ncnn.LowerClamp>();
            p.Add<Passes.Rules.Ncnn.LowerConcat>();
            p.Add<Passes.Rules.Ncnn.LowerConv>();
            p.Add<Passes.Rules.Ncnn.LowerCumsum>();
            p.Add<Passes.Rules.Ncnn.LowerElu>();

            p.Add<Passes.Rules.Ncnn.LowerErf>();
            p.Add<Passes.Rules.Ncnn.LowerHardSigmoid>();
            p.Add<Passes.Rules.Ncnn.LowerHardSwish>();
            p.Add<Passes.Rules.Ncnn.LowerInstanceNorm>();
            p.Add<Passes.Rules.Ncnn.LowerLayerNorm>();
            p.Add<Passes.Rules.Ncnn.LowerLRN>();
            p.Add<Passes.Rules.Ncnn.LowerLSTM>();
            p.Add<Passes.Rules.Ncnn.LowerPadding>();
            p.Add<Passes.Rules.Ncnn.LowerPooling>();
            p.Add<Passes.Rules.Ncnn.LowerPReLU>();
            p.Add<Passes.Rules.Ncnn.LowerReduction>();

            p.Add<Passes.Rules.Ncnn.LowerSELU>();
            p.Add<Passes.Rules.Ncnn.LowerSigmoid>();
            p.Add<Passes.Rules.Ncnn.LowerCrop>();
            p.Add<Passes.Rules.Ncnn.LowerSoftplus>();
            p.Add<Passes.Rules.Ncnn.LowerSlice>();
            p.Add<Passes.Rules.Ncnn.LowerTile>();
            p.Add<Passes.Rules.Ncnn.LowerPermute>();
            p.Add<Passes.Rules.Ncnn.LowerMatmul>();
            p.Add<Passes.Rules.Ncnn.LowerConvTranspose>();
            p.Add<Passes.Rules.Ncnn.LowerCast>();
            p.Add<Passes.Rules.Ncnn.LowerGELU>();

            p.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
            p.Add<Passes.Rules.Neutral.FoldNopReshape>();
            p.Add<Passes.Rules.Ncnn.LowerReshape>();

            // p.Add<Passes.Rules.Ncnn.LowerDequantize>(); // ncnn dequantize int to float.
        });

        passManager.AddWithName<DataflowPass>("RemoveGlueOp").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldSqueezeUnsqueeze>();
            p.Add<Passes.Rules.Neutral.FoldUnsqueezeSqueeze>();
        });

        passManager.AddWithName<DataflowPass>("RemoveSingleSqueezeAndUnsqueeze").Configure(p =>
        {
            p.Add<Passes.Rules.Ncnn.LowerSqueeze>();
            p.Add<Passes.Rules.Ncnn.LowerUnsqueeze>();
        });
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
    }

    public void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options)
    {
        passManager.Add<FusionToFunctionPass>();
    }

    /// <inheritdoc/>
    public IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options)
    {
        if (moduleKind == Callable.StackVMModuleKind)
        {
            return new StackVMModuleBuilder();
        }
        else if (moduleKind == "ncnn")
        {
            return new NcnnModuleBuilder();
        }
        else
        {
            throw new NotSupportedException($"{moduleKind} module is not supported.");
        }
    }
}
