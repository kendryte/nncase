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
using Nncase.Quantization;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Nncase.Transform.Rules.Lower;
using Nncase.Utilities;

namespace Nncase.Compiler;

internal class Compiler : ICompiler
{
    private readonly CompileSession _compileSession;
    private readonly IModelBuilder _modelBuilder;
    private readonly IDumpper _dumpper;
    private IRModule? _module;

    public Compiler(CompileSession compileSession, IModelBuilder modelBuilder, IDumpperFactory dumpperFactory)
    {
        _compileSession = compileSession;
        _modelBuilder = modelBuilder;
        _dumpper = dumpperFactory.Root;
    }

    public IRModule Module => _module ?? throw new InvalidOperationException("Module has not been imported");

    public async Task<IRModule> ImportModuleAsync(Stream content)
    {
        var module = ImportModel(content);
        if (_dumpper.IsEnabled(DumpFlags.Compile))
        {
            _dumpper.DumpModule(module, "IRImport");
        }

        await RunPassAsync(pmg => pmg.Add<ShapeInferPass>(), "ShapeInferAfterImport");

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        return module;
    }

    public void TargetIndependentPass(IPassManager passManager)
    {
        var quantMode = _compileSession.CompileOptions.QuantizeOptions.ModelQuantMode;
        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.Add<EGraphPass>()
                .Configure(p =>
                {
                    p.Name = "NeutralOptimize";
                    p.Add<Transform.Rules.Neutral.FoldConstCall>()
                     .Add<Transform.Rules.Neutral.FoldNopTranspose>()
                     .Add<Transform.Rules.Neutral.FoldTwoTransposes>()
                     .Add<Transform.Rules.Neutral.CombineTransposeUnary>()
                     .Add<Transform.Rules.Neutral.CombineTransposePad>()
                     .Add<Transform.Rules.Neutral.CombineTransposeBinary>()
                     .Add<Transform.Rules.Neutral.CombineTransposeReduce>()
                     .Add<Transform.Rules.Neutral.CombineTransposeActivations>()
                     .Add<Transform.Rules.Neutral.CombinePadTranspose>()
                     .Add<Transform.Rules.Neutral.FoldNopPad>()
                     .Add<Transform.Rules.Neutral.FoldConv2DPads>()
                     .Add<Transform.Rules.Neutral.FoldReduceWindow2DPads>();
                });
        }

        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.Add<EGraphPass>()
                .Configure(p =>
                {
                    p.Name = "AddRangeOfMarker";
                    p.Add<Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2D>()
                     .Add<Transform.Rules.Neutral.AddRangeOfAndMarkerToMatMul>();
                });
            passManager.Add<EGraphPassWithQuantize>().Configure(p => p.Name = "AssignRanges");
        }
    }

    public async Task CompileAsync()
    {
        var target = _compileSession.Target;
        await RunPassAsync(p => TargetIndependentPass(p), "TargetIndependentPass");
        await RunPassAsync(p => target.RegisterTargetDependentPass(p, _compileSession.CompileOptions), "TargetIndependentPass");

        if (_compileSession.CompileOptions.QuantizeOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            await RunPassAsync(p => target.RegisterQuantizePass(p, _compileSession.CompileOptions), "QuantizePass");
            await RunPassAsync(p => target.RegisterTargetDependentAfterQuantPass(p, _compileSession.CompileOptions), "TargetDependentAfterQuantPass");
            await RunPassAsync(
                pmgr => pmgr.Add<DataflowPass>().Configure(p =>
                {
                    p.Name = "ClearMarker";
                    p.Add<RemoveMarker>();
                }),
                "RemoveMarker");
        }

        // fold constant
        await RunPassAsync(p => p.Add<ShapeInferPass>(), "ShapeInferAfterCompile");
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
        var pmgr = _compileSession.CreatePassManager(name);
        register(pmgr);
        _module = await pmgr.RunAsync(Module).ConfigureAwait(false);

        if (_dumpper.IsEnabled(DumpFlags.Compile))
        {
            _dumpper.DumpModule(_module, name);
        }
    }
}
