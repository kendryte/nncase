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
            passManager.Add(new EGraphPass("1_NeutralOptimize")
            {
              new Transform.Rules.Neutral.FoldConstCall(),
              new Transform.Rules.Neutral.FoldNopTranspose(),
              new Transform.Rules.Neutral.FoldTwoTransposes(),
              new Transform.Rules.Neutral.CombineTransposeUnary(),
              new Transform.Rules.Neutral.CombineTransposePad(),
              new Transform.Rules.Neutral.CombinePadTranspose(),
              new Transform.Rules.Neutral.CombineTransposeBinary(),
              new Transform.Rules.Neutral.CombineTransposeConstBinary(),
              new Transform.Rules.Neutral.CombineTransposeReduce(),
              new Transform.Rules.Neutral.CombineTransposeActivations(),
              new Transform.Rules.Neutral.CombineActivationsTranspose(),
              new Transform.Rules.Neutral.FoldNopPad(),
              new Transform.Rules.Neutral.FoldConv2DPads(),
              new Transform.Rules.Neutral.FoldReduceWindow2DPads(),
            });
        }

        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.Add(new DataflowPass("2_AddRangeOfMarker")
            {
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2D(),
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToMatMul(),
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToReduceWindow2D(),
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2DTranspose(),
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToBinary(),
            });
            passManager.Add(new Quantization.EGraphPassWithQuantize("3_AssignRanges", _compileOptions.QuantizeOptions!));
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
