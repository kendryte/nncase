// Copyright (c) Canaan Inc. All rights reserved.
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
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
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
        passManager.AddWithName<DataflowPass>("TargetDependentNeutralOptimize").Configure(p =>
        {
            // p.Add<BroadcastTransposeOutputNames>();
            // p.Add<BroadcastReshapeOutputNames>();
            // p.Add<BroadcastNopPadOutputNames>();
            // p.Add<IntegralPromotion>();
            // p.Add<FoldConstCall>();
            // p.Add<FoldShapeOf>();
            // p.Add<TransposeToReshape>();
            // p.Add<ExpandToBroadcast>();
            // p.Add<MatMulToConv2DWithMarker>();
            // p.Add<BroadcastMatMulToConv2DWithMarker>();
            //s p.Add<MatMulToConv2D>();
            // p.Add<BroadcastMatMulToConv2D>();
            // p.Add<BroadcastMatMul>();
            p.Add<ReshapeBatchMatmul>();
            // p.Add<SplitBatchMatMul>();
            // p.Add<BatchNormToBinary>();
            p.Add<FoldTwoReshapes>();
            p.Add<FoldNopReshape>();
            // p.Add<FoldTwoReduce>();
            // p.Add<CombineBinaryReshape>();
            // p.Add<CombineConstBinaryReshape>();
            // p.Add<CombineUnaryReshape>();
            // p.Add<CombineActivationsReshape>();
            // p.Add<FoldNopBroadcast>();
        });
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
    }

    public void RegisterTargetDependentBeforeCodeGen(IPassManager passManager, CompileOptions options)
    {
#if false
        passManager.AddWithName<DataflowPass>("QuantizeWeights").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FloatConstToBFloat16>();
        });
#endif
    }

    /// <inheritdoc/>
    public IModuleBuilder CreateModuleBuilder(string moduleKind, CompileOptions options)
    {
        if (moduleKind == Callable.StackVMModuleKind)
        {
            return new StackVMModuleBuilder();
        }
        else
        {
            throw new NotSupportedException($"{moduleKind} module is not supported.");
        }
    }
}
