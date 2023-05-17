// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Transforms;
using Nncase.Quantization;
using Nncase.Utilities;

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

    public async Task<IRModule> ImportModuleAsync(Stream content)
    {
        var module = ImportModel(content);
        if (_dumpper.IsEnabled(DumpFlags.Compile))
        {
            _dumpper.DumpModule(module, "IRImport");
        }

        await RunPassAsync(pmg => BroadcastOutputNamesAfterImportPass(pmg), "BroadcastOutputNamesAfterImport");
        await RunPassAsync(pmg => pmg.Add<ShapeInferPass>(), "ShapeInferAfterImport");

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        return module;
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

    public void TargetIndependentPass(IPassManager passManager)
    {
        var quantMode = _compileSession.CompileOptions.QuantizeOptions.ModelQuantMode;
        passManager.AddWithName<DataflowPass>("SqueezeShape").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.SqueezeTransposeShape>();
            p.Add<Passes.Rules.Neutral.Squeeze5DTranspose>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern1>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern2>();
            p.Add<Passes.Rules.Neutral.FoldLayerNormPattern3>();
            p.Add<Passes.Rules.Neutral.FoldGeluWithScale>();
            p.Add<Passes.Rules.Neutral.FoldGeneralGelu>();
            p.Add<Passes.Rules.Neutral.FoldSwishPattern1>();
            p.Add<Passes.Rules.Neutral.FoldSwishPattern2>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish1>();
            p.Add<Passes.Rules.Neutral.FoldHardSwish2>();
            p.Add<Passes.Rules.Neutral.FocusFull>();
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
        _compileSession.Target.RegisterTargetInDependentPass(passManager, _compileSession.CompileOptions);

        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("AddRangeOfMarker").Configure(p =>
            {
                p.Add<Passes.Rules.Neutral.AddRangeOfAndMarker>();
            });
            passManager.AddWithName<EGraphPassWithQuantize>("AssignRanges");
        }
    }

    public async Task CompileAsync()
    {
        var target = _compileSession.Target;
        await RunPassAsync(p => TargetIndependentPass(p), "TargetIndependentPass");
        await RunPassAsync(p => target.RegisterTargetDependentPass(p, _compileSession.CompileOptions), "TargetDependentPass");
        await RunPassAsync(p => target.RegisterQuantizePass(p, _compileSession.CompileOptions), "QuantizePass");
        await RunPassAsync(p => target.RegisterTargetDependentAfterQuantPass(p, _compileSession.CompileOptions), "TargetDependentAfterQuantPass");
        await RunPassAsync(p => target.RegisterTargetDependentBeforeCodeGen(p, _compileSession.CompileOptions), "TargetDependentBeforeCodeGen");
    }

    public void Gencode(Stream output)
    {
        var linkedModel = _modelBuilder.Build(Module);
        linkedModel.Serialize(output);
    }

    private IRModule ImportModel(Stream content)
    {
        _module = _compileSession.CompileOptions.InputFormat switch
        {
            "tflite" => Importers.ImportTFLite(content, _compileSession),
            "onnx" => Importers.ImportOnnx(content, _compileSession),
            var inputFormat => throw new NotImplementedException($"Not Implement {inputFormat} Importer!"),
        };
        return _module;
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
