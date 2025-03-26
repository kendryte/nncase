// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Security.AccessControl;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Distributed;
using Nncase.Passes.Mutators;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Passes.Rules.WithMarker;
using Nncase.Passes.Transforms;
using Nncase.Quantization;
using CombinePadTranspose = Nncase.Passes.Rules.WithMarker.CombinePadTranspose;
using CombineReshapePad = Nncase.Passes.Rules.Neutral.CombineReshapePad;
using FoldConstCall = Nncase.Passes.Rules.Neutral.FoldConstCall;

namespace Nncase.Compiler;

internal class Compiler : ICompiler
{
    private readonly CompileSession _compileSession;
    private readonly IModelBuilder _modelBuilder;
    private IRModule? _module;
    private int _runPassCount;

    public Compiler(CompileSession compileSession, IModelBuilder modelBuilder)
    {
        _compileSession = compileSession;
        _modelBuilder = modelBuilder;
        _runPassCount = 0;
    }

    public IRModule Module => _module ?? throw new InvalidOperationException("Module has not been imported");

    /// <inheritdoc/>
    public void ImportIRModule(IRModule module) => _module = module;

    public async Task<IRModule> ImportTFLiteModuleAsync(Stream content)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportTFLite(content, _compileSession);
        return await InitializeModuleAsync(module);
    }

    public async Task<IRModule> ImportOnnxModuleAsync(Stream content)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportOnnx(content, _compileSession);
        return await InitializeModuleAsync(module);
    }

    public async Task<IRModule> ImportNcnnModuleAsync(Stream ncnnParam, Stream ncnnBin)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var module = Importers.ImportNcnn(ncnnParam, ncnnBin, _compileSession);
        return await InitializeModuleAsync(module);
    }

    public void RegisterPreAndPostProcess(IPassManager passManager)
    {
        passManager.Add<AddPreProcess>();
        passManager.Add<AddPostProcess>();
        passManager.AddWithName<DataflowPass>("FoldNopBinary").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldNopBinary>();
        });
    }

    public void ProcessAfterImportPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("FoldQuantDeQuant").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldQuantDeQuant>();
        });
        passManager.AddWithName<DataflowPass>("BroadcastOutputNamesAfterImportPass").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.BroadcastTransposeOutputNames>();
            p.Add<Passes.Rules.Neutral.BroadcastReshapeOutputNames>();
            p.Add<Passes.Rules.Neutral.BroadcastNopPadOutputNames>();
        });
        passManager.Add<ShapeInferPass>();
        RegisterPreAndPostProcess(passManager);
    }

    public void TargetIndependentPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("NormAxisAndShape").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.ReshapeMatMul>();
            p.Add<Passes.Rules.Neutral.NormAxisGather>();
            p.Add<Passes.Rules.Neutral.NormAxisConcat>();
            p.Add<Passes.Rules.Neutral.NormAxisReduce>();
            p.Add<Passes.Rules.Neutral.NormAxisReshape>();
            p.Add<Passes.Rules.Neutral.NormAxisReduceArg>();
            p.Add<Passes.Rules.Neutral.NormAxisSlice>();
            p.Add<Passes.Rules.Neutral.SqueezeTransposeShape>();
            p.Add<Passes.Rules.Neutral.Squeeze5DTranspose>();
            p.Add<Passes.Rules.Neutral.SqueezeBinaryShape>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern1>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern2>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern3>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern4>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern5>();
            p.Add<Passes.Rules.Neutral.ConvertLayerNormChannelFirstToLast>();
            p.Add<Passes.Rules.Neutral.FoldGeluWithScale>();
            p.Add<Passes.Rules.Neutral.FoldGeneralGelu>();
            p.Add<Passes.Rules.Neutral.FoldSwishPattern1>();
            p.Add<Passes.Rules.Neutral.FoldSwishPattern2>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish1>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish2>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish3>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish4>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish5>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.FocusFull>();
            p.Add<Passes.Rules.Neutral.ReshapeMatMul>();
            p.Add<Passes.Rules.Neutral.ReshapeExpand>();
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldShapeOf>();
            p.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.FoldNopBinary>();
            p.Add<Passes.Rules.Neutral.FoldNopCast>();
            p.Add<Passes.Rules.Neutral.FoldNopReshape>();
            p.Add<Passes.Rules.Neutral.FoldNopSlice>();
            p.Add<Passes.Rules.Neutral.FoldSqueezeUnsqueeze>();
            p.Add<Passes.Rules.Neutral.FoldUnsqueezeSqueeze>();
            p.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
            p.Add<Passes.Rules.Neutral.FoldNopClamp>();
            p.Add<Passes.Rules.ShapeBucket.FoldRepeatMarker>();
            p.Add<Passes.Rules.Neutral.SqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.UnSqueezeToReshape>();
            p.Add<Passes.Rules.ShapeExpr.GatherToGetItem>();
            p.Add<Passes.Rules.ShapeExpr.FoldGetItemShapeOf>();
            p.Add<Passes.Rules.Neutral.FoldGetItemConcat>();
            p.Add<Passes.Rules.Neutral.FoldGetItemReshape>();
            p.Add<Passes.Rules.Neutral.FoldIf>();
            p.Add<Passes.Rules.Neutral.FoldNopReduce>();
            p.Add<Passes.Rules.Neutral.SliceToGetItem>();
            p.Add<Passes.Rules.Neutral.FoldTwoPads>();
            p.Add<Passes.Rules.Neutral.SwapBinaryArgs>();
            p.Add<Passes.Rules.Neutral.FoldDilatedConv2D>();
            p.Add<Passes.Rules.Neutral.PowOf2ToSquare>();
            p.Add<Passes.Rules.Neutral.ScalarConstToTensor>();
        });

        // Decompose complex ops
        passManager.AddWithName<DataflowPass>("DecomposeComplexOps").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.SwapBinaryArgs>();
            p.Add<Passes.Rules.Neutral.DecomposeSoftmax>();
            p.Add<Passes.Rules.Neutral.DecomposeLayerNorm>();
            p.Add<Passes.Rules.Neutral.DecomposeInstanceNorm>();
            p.Add<Passes.Rules.Neutral.DecomposeGelu>();
            p.Add<Passes.Rules.Neutral.ScalarConstToTensor>();
        });

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();

        passManager.AddWithName<EGraphRulesPass>("NeutralOptimizeTranspose").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldNopTranspose>();
            p.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
            p.Add<FoldRepeatMarker>();
            p.Add<Passes.Rules.WithMarker.FoldTransposeActTranspose>();
            p.Add<Passes.Rules.WithMarker.FoldTransposeBinaryActTranspose>();
            p.Add<Passes.Rules.WithMarker.CombineReshapePad>();
            p.Add<Passes.Rules.WithMarker.CombinePadTranspose>();
            p.Add<Passes.Rules.Neutral.CombineUnaryTranspose>();
            if (_compileSession.CompileOptions.ShapeBucketOptions.Enable)
            {
                p.Add<Passes.Rules.WithMarker.CombineTransposePad>();
            }
            else
            {
                p.Add<Passes.Rules.Neutral.CombineTransposePad>();
            }

            p.Add<Passes.Rules.Neutral.CombinePadTranspose>();
            p.Add<Passes.Rules.Neutral.CombineBinaryTranspose>();
            p.Add<Passes.Rules.Neutral.CombineConstBinaryTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTransposeConstBinary>();
            p.Add<Passes.Rules.Neutral.CombineTransposeReduce>();
            p.Add<Passes.Rules.Neutral.CombineTransposeActivations>();
            p.Add<Passes.Rules.Neutral.CombineActivationsTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTransposeConcat>();
            p.Add<Passes.Rules.Neutral.CombineBinaryReshape>();
            p.Add<Passes.Rules.Neutral.CombineConstBinaryReshape>();
            p.Add<Passes.Rules.Neutral.CombineUnaryReshape>();
            p.Add<Passes.Rules.Neutral.CombineActivationsReshape>();
            p.Add<Passes.Rules.Neutral.CombineReshapePad>();
            p.Add<Passes.Rules.Neutral.CombineReshapeTranspose>();
            p.Add<Passes.Rules.Neutral.CombineTransposeReshape>();
            p.Add<Passes.Rules.Neutral.FoldNopPad>();
            p.Add<Passes.Rules.Neutral.FoldConv2DPads>();
            p.Add<Passes.Rules.Neutral.FuseClampConv2D>();
            p.Add<Passes.Rules.Neutral.FoldReduceWindow2DPads>();
            p.Add<Passes.Rules.Neutral.SqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.UnSqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.TransposeToReshape>();
            p.Add<Passes.Rules.Neutral.FlattenToReshape>();
            p.Add<Passes.Rules.Neutral.ReshapeToTranspose>();
            p.Add<Passes.Rules.Neutral.FoldNopReshape>();
            p.Add<Passes.Rules.Neutral.FoldTwoReshapes>();
            p.Add<Passes.Rules.Neutral.FoldReshapeBinaryConstReshape>();
            p.Add<Passes.Rules.Neutral.ReluToClamp>();
            p.Add<Passes.Rules.Neutral.Relu6ToClamp>();
            p.Add<Passes.Rules.Neutral.FoldNopSlice>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.SpaceToBatchToPad>();
            p.Add<Passes.Rules.Neutral.FoldConv2DAddMul>();
        });

        _compileSession.Target.RegisterTargetInDependentPass(passManager, _compileSession.CompileOptions);

        passManager.AddWithName<DataflowPass>("BroadcastMarker").Configure(p =>
        {
            p.Add<FoldTransposeActTranspose>();
            p.Add<BroadcastInputMarker>();
            p.Add<BroadcastOutputMarker>();
        });
    }

    public void QuantizePass(IPassManager passManager)
    {
        var options = _compileSession.CompileOptions;
        _compileSession.Target.RegisterQuantizePass(passManager, options);
        if (options.QuantizeOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("RemoveMarker").Configure(p =>
            {
                p.Add<Passes.Rules.Lower.RemoveMarker>();
            });
        }
    }

    public void ModulePartitionPass(IPassManager passManager)
    {
        foreach (var moduleCompiler in _compileSession.Target.ModuleCompilers)
        {
            passManager.AddWithName<ModulePartitionPass>($"ModulePartition_{moduleCompiler.ModuleKind}", moduleCompiler);
        }

        passManager.Add<RemoveUnusedFunctions>();
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void AutoPackingPass(IPassManager passManager)
    {
        var target = _compileSession.Target;
        passManager.AddWithName<EGraphRulesPass>("AutoPacking").Configure(p =>
        {
            target.RegisterAutoPackingRules(p, _compileSession.CompileOptions);
        });

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void AutoDistributedPass(IPassManager passManager)
    {
        foreach (var moduleCompiler in _compileSession.Target.ModuleCompilers)
        {
            passManager.AddWithName<AutoDistributedPass>($"AutoDistributed_{moduleCompiler.ModuleKind}", true, moduleCompiler.ModuleKind);
        }

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }

    public void TIRPass(IPassManager passManager)
    {
        var target = _compileSession.Target;
        target.RegisterTIRSelectionPass(passManager, _compileSession.CompileOptions);
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();

        passManager.AddWithName<DataflowPass>("AffineSelection").Configure(p =>
        {
            target.RegisterAffineSelectionRules(p, _compileSession.CompileOptions);
        });

        foreach (var moduleCompiler in _compileSession.Target.ModuleCompilers)
        {
            passManager.AddWithName<AutoTilePass>($"AutoTiling_{moduleCompiler.ModuleKind}", moduleCompiler.ModuleKind);
        }

        passManager.Add<AddFunctionToModule>();
        passManager.Add<BufferizePass>();

        passManager.AddWithName<PrimFuncPass>("Optimize").Configure(p =>
        {
            p.Add<Passes.Mutators.UnFoldBlock>();
            p.Add<Passes.Mutators.FlattenSequential>();
            p.Add<Passes.Mutators.TailLoopStripping>();
            p.Add<Passes.Mutators.FoldConstCall>();
            p.Add<Passes.Mutators.FlattenBuffer>();
            p.Add<Passes.Mutators.RemoveNop>();
        });
    }

    public async Task CompileAsync(IProgress<int>? progress = null, CancellationToken token = default)
    {
        Task RunPassAsync(Action<IPassManager> register, string name)
        {
            return this.RunPassAsync(register, name, progress, token);
        }

        var target = _compileSession.Target;
        using var scope = new CompileSessionScope(_compileSession);
        await RunPassAsync(TargetIndependentPass, "TargetIndependentPass");
        await RunPassAsync(TargetIndependQuantPass, "TargetIndependentQuantPass");

        await RunPassAsync(
            p => target.RegisterTargetDependentPass(p, _compileSession.CompileOptions),
            "TargetDependentPass");
        await RunPassAsync(QuantizePass, "QuantizePass");

        await RunPassAsync(ModulePartitionPass, "ModulePartitionPass");
        await RunPassAsync(AutoPackingPass, "AutoPackingPass");
        await RunPassAsync(AutoDistributedPass, "AutoDistributedPass");

        await RunPassAsync(TIRPass, "TIRPass");

        await RunPassAsync(
            p =>
            {
                // target.RegisterTargetDependentBeforeCodeGen(p, _compileSession.CompileOptions);
                p.Add<ReplaceDimVarWithShapeOfPass>();
            },
            "TargetDependentBeforeCodeGen");
        if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
        {
            DumpScope.Current.DumpModule(_module!, "ModuleAfterCompile");
        }
    }

    public void Gencode(Stream output)
    {
        using var scope = new CompileSessionScope(_compileSession);
        var linkedModel = _modelBuilder.Build(Module);
        linkedModel.Serialize(output);
    }

    private async Task<IRModule> InitializeModuleAsync(IRModule module)
    {
        _module = module;

        if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
        {
            DumpScope.Current.DumpModule(module, "IRImport");
        }

        var preprocess_option = _compileSession.CompileOptions;

        await RunPassAsync(pmg => ProcessAfterImportPass(pmg), "ProcessAfterImportPass");

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        if (!inferSucc)
        {
            throw new InvalidOperationException("Type inference failed");
        }

        return module;
    }

    private void TargetIndependQuantPass(IPassManager passManager)
    {
        var quantMode = _compileSession.CompileOptions.QuantizeOptions.ModelQuantMode;
        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("AddRangeOfMarker").Configure(p =>
            {
                p.Add<Passes.Rules.Neutral.AddRangeOfAndMarker>();
            });
            passManager.AddWithName<EGraphPassWithQuantize>("AssignRanges");
        }
    }

    private async Task RunPassAsync(Action<IPassManager> register, string name, IProgress<int>? progress = null, CancellationToken token = default)
    {
        var newName = $"{_runPassCount++}_" + name;
        var pmgr = _compileSession.CreatePassManager(newName);
        register(pmgr);
        _module = await pmgr.RunAsync(Module).ConfigureAwait(false);

        if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
        {
            DumpScope.Current.DumpModule(_module, newName);
            DumpScope.Current.DumpDotIR(_module.Entry!, newName);
        }

        progress?.Report(_runPassCount);
        token.ThrowIfCancellationRequested();
    }
}
