// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Security.AccessControl;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Passes.Transforms;
using Nncase.Quantization;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketRegister;
using FoldConstCall = Nncase.Passes.Rules.Neutral.FoldConstCall;

namespace Nncase.Compiler;

internal class Compiler : ICompiler
{
    private readonly CompileSession _compileSession;
    private readonly IModelBuilder _modelBuilder;
    private readonly IDumpper _dumpper;
    private IRModule? _module;
    private int _runPassCount;

    public Compiler(CompileSession compileSession, IModelBuilder modelBuilder, IDumpperFactory dumpperFactory)
    {
        _compileSession = compileSession;
        _modelBuilder = modelBuilder;
        _dumpper = dumpperFactory.Root;
        _runPassCount = 0;
    }

    public IRModule Module => _module ?? throw new InvalidOperationException("Module has not been imported");

    /// <inheritdoc/>
    public void ImportIRModule(IRModule module) => _module = module;

    public Task<IRModule> ImportTFLiteModuleAsync(Stream content)
    {
        var module = Importers.ImportTFLite(content, _compileSession);
        return InitializeModuleAsync(module);
    }

    public Task<IRModule> ImportOnnxModuleAsync(Stream content)
    {
        var module = Importers.ImportOnnx(content, _compileSession);
        return InitializeModuleAsync(module);
    }

    public Task<IRModule> ImportNcnnModuleAsync(Stream ncnnParam, Stream ncnnBin)
    {
        var module = Importers.ImportNcnn(ncnnParam, ncnnBin, _compileSession);
        return InitializeModuleAsync(module);
    }

    public void BroadcastOutputNamesAfterImportPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("BroadcastOutputNamesAfterImportPass").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.BroadcastTransposeOutputNames>();
            p.Add<Passes.Rules.Neutral.BroadcastReshapeOutputNames>();
            p.Add<Passes.Rules.Neutral.BroadcastNopPadOutputNames>();
        });
    }

    public void AddPreAndPostProcess(IPassManager passManager)
    {
        passManager.Add<AddPreProcess>();
        passManager.Add<AddPostProcess>();
        passManager.AddWithName<DataflowPass>("FoldNopBinary").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldNopBinary>();
        });
    }

    public void TargetIndependentPass(IPassManager passManager)
    {
        passManager.AddWithName<DataflowPass>("ReshapeMatMul").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.ReshapeMatMul>();
        });

        passManager.AddWithName<DataflowPass>("SqueezeShape").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.SqueezeTransposeShape>();
            p.Add<Passes.Rules.Neutral.Squeeze5DTranspose>();
            p.Add<Passes.Rules.Neutral.SqueezeBinaryShape>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern1>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern2>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern3>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern4>();
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
            p.Add<Passes.Rules.Neutral.SplitSpaceToBatch>();
            p.Add<Passes.Rules.Neutral.SplitBatchToSpace>();
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
            p.Add<Passes.Rules.Neutral.SqueezeToReshape>();
            p.Add<Passes.Rules.Neutral.UnSqueezeToReshape>();
            p.Add<Passes.Rules.ShapeExpr.GatherToGetItem>();
            p.Add<Passes.Rules.ShapeExpr.FoldGetItemShapeOf>();
            p.Add<Passes.Rules.Neutral.FoldNopReduce>();
            p.Add<Passes.Rules.Neutral.SliceToGetItem>();
            p.Add<Passes.Rules.Neutral.FoldTwoPads>();
        });

        passManager.AddWithName<EGraphRulesPass>("NeutralOptimizeTranspose").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.FoldConstCall>();
            p.Add<Passes.Rules.Neutral.FoldNopTranspose>();
            p.Add<Passes.Rules.Neutral.FoldTwoTransposes>();
            p.Add<Passes.Rules.Neutral.CombineTransposeUnary>();
            p.Add<Passes.Rules.Neutral.CombineTransposePad>();
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
            p.Add<Passes.Rules.Neutral.ReluToClamp>();
            p.Add<Passes.Rules.Neutral.Relu6ToClamp>();
            p.Add<Passes.Rules.Neutral.FoldNopSlice>();
            p.Add<Passes.Rules.Neutral.FoldTwoSlices>();
            p.Add<Passes.Rules.Neutral.SpaceToBatchToPad>();
        });

        _compileSession.Target.RegisterTargetInDependentPass(passManager, _compileSession.CompileOptions);

        passManager.AddWithName<DataflowPass>("BroadcastMarker").Configure(p =>
        {
            p.Add<BroadcastInputMarker>();
            p.Add<BroadcastOutputMarker>();
        });

        // passManager.AddWithName<EGraphPass>("NeutralOptimizeClamp").Configure(p =>
        // {
        //     p.Add<Passes.Rules.Neutral.FoldConstCall>();
        //     p.Add<Passes.Rules.Neutral.FoldConv2DAddMul>();
        //     p.Add<Passes.Rules.Neutral.ReluToClamp>();
        //     p.Add<Passes.Rules.Neutral.Relu6ToClamp>();
        //     p.Add<Passes.Rules.Neutral.CombineClampAdd>();
        //     p.Add<Passes.Rules.Neutral.CombineClampMul>();
        //     p.Add<Passes.Rules.Neutral.FoldNopClamp>();
        // });
    }

    public void RegisterShapeBucket(IPassManager p)
    {
        var options = _compileSession.CompileOptions.ShapeBucketOptions;
        if (!options.Enable)
        {
            return;
        }

        var singleVar = options.VarMap.Values.SelectMany(x => x).OfType<Var>().ToHashSet().Count <= 1;
        CheckShapeBucketOptions(options);

        if (HasNotBucketOp(_module!.Entry!) || !singleVar)
        {
            ToFusion(p);
            MergeOp(p, true);
            LostToFusion(p, singleVar);
            MergeOp(p, true);
            ClearMarker(p);
            MergeFusion(p, singleVar, true);
            Bucket(p);
            Rebuild(p, singleVar);
            Simplify(p);
        }
        else
        {
            p.AddWithName<FullBucket>("FullBucket");
        }
    }

    public void ClearFixShape(IPassManager p)
    {
        if (!_compileSession.CompileOptions.ShapeBucketOptions.Enable)
        {
            return;
        }

        p.AddWithName<DataflowPass>("ClearUnused").Configure(c =>
        {
            c.Add<FoldFixShape>();
            c.Add<ClearRequire>();
        });
    }

    public void DoProcessing(IProgress<int> progress)
    {
        var maxPassCount = 0;
        while (_runPassCount < maxPassCount)
        {
            Thread.Sleep(1000);
            progress?.Report(_runPassCount);
        }
    }

    public async Task CompileAsync()
    {
        Console.WriteLine("Run");
        var target = _compileSession.Target;
        await RunPassAsync(p => TargetIndependentPass(p), "TargetIndependentPass");
        await RunPassAsync(p => RegisterTargetIndependQuantPass(p), "TargetIndependentQuantPass");
        if (_compileSession.CompileOptions.ShapeBucketOptions.Enable)
        {
            await RunPassAsync(p => RegisterShapeBucket(p), "ShapeBucket");
            await RunPassAsync(p => TargetIndependentPass(p), "TargetIndependentPass");
        }

        Console.WriteLine("Target independent");
        await RunPassAsync(
            p => target.RegisterTargetDependentPass(p, _compileSession.CompileOptions),
            "TargetDependentPass");
        await RunPassAsync(p => target.RegisterQuantizePass(p, _compileSession.CompileOptions), "QuantizePass");
        await RunPassAsync(
            p => target.RegisterTargetDependentAfterQuantPass(p, _compileSession.CompileOptions),
            "TargetDependentAfterQuantPass");
        await RunPassAsync(p => ClearFixShape(p), "ClearFixShape");
        await RunPassAsync(
            p => target.RegisterTargetDependentBeforeCodeGen(p, _compileSession.CompileOptions),
            "TargetDependentBeforeCodeGen");
        if (_dumpper.IsEnabled(DumpFlags.Compile))
        {
            DumpScope.Current.DumpModule(_module!, "ModuleAfterCompile");
        }
    }

    public void Gencode(Stream output)
    {
        var linkedModel = _modelBuilder.Build(Module);
        linkedModel.Serialize(output);
    }

    private async Task<IRModule> InitializeModuleAsync(IRModule module)
    {
        _module = module;

        if (_dumpper.IsEnabled(DumpFlags.Compile))
        {
            _dumpper.DumpModule(module, "IRImport");
        }

        var preprocess_option = _compileSession.CompileOptions;

        await RunPassAsync(pmg => BroadcastOutputNamesAfterImportPass(pmg), "BroadcastOutputNamesAfterImport");
        await RunPassAsync(pmg => pmg.Add<ShapeInferPass>(), "ShapeInferAfterImport");
        await RunPassAsync(pmg => AddPreAndPostProcess(pmg), "AddPreAndPostProcessAfterImport");

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        return module;
    }

    private void RegisterTargetIndependQuantPass(IPassManager passManager)
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

    private async Task RunPassAsync(Action<IPassManager> register, string name)
    {
        var newName = $"{_runPassCount++}_" + name;
        var pmgr = _compileSession.CreatePassManager(newName);
        register(pmgr);
        _module = await pmgr.RunAsync(Module).ConfigureAwait(false);

        if (_dumpper.IsEnabled(DumpFlags.Compile))
        {
            _dumpper.DumpModule(_module, newName);
            _dumpper.DumpDotIR(_module.Entry!, newName);
        }
    }
}
