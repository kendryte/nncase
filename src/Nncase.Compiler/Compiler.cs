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

    /// <inheritdoc/>
    public void ImportIRModule(IRModule module) => _module = module;

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
            passManager.AddWithName<EGraphPass>("NeutralOptimizeTranspose").Configure(p =>
            {
                p.Add<Transform.Rules.Neutral.FoldConstCall>();
                p.Add<Transform.Rules.Neutral.FoldNopTranspose>();
                p.Add<Transform.Rules.Neutral.FoldTwoTransposes>();
                p.Add<Transform.Rules.Neutral.CombineTransposeUnary>();
                p.Add<Transform.Rules.Neutral.CombineTransposePad>();
                p.Add<Transform.Rules.Neutral.CombinePadTranspose>();
                p.Add<Transform.Rules.Neutral.CombineBinaryTranspose>();
                p.Add<Transform.Rules.Neutral.CombineConstBinaryTranspose>();
                p.Add<Transform.Rules.Neutral.CombineTransposeConstBinary>();
                p.Add<Transform.Rules.Neutral.CombineTransposeReduce>();
                p.Add<Transform.Rules.Neutral.CombineTransposeActivations>();
                p.Add<Transform.Rules.Neutral.CombineActivationsTranspose>();
                p.Add<Transform.Rules.Neutral.FoldNopPad>();
                p.Add<Transform.Rules.Neutral.FoldConv2DPads>();
                p.Add<Transform.Rules.Neutral.FoldReduceWindow2DPads>();
            });

            // passManager.AddWithName<EGraphPass>("NeutralOptimizeClamp").Configure(p =>
            // {
            //     p.Add<Transform.Rules.Neutral.FoldConstCall>();
            //     p.Add<Transform.Rules.Neutral.FoldConv2DAddMul>();
            //     p.Add<Transform.Rules.Neutral.ReluToClamp>();
            //     p.Add<Transform.Rules.Neutral.Relu6ToClamp>();
            //     p.Add<Transform.Rules.Neutral.CombineClampAdd>();
            //     p.Add<Transform.Rules.Neutral.CombineClampMul>();
            //     p.Add<Transform.Rules.Neutral.FoldNopClamp>();
            // });
        }

        if (quantMode == ModelQuantMode.UsePTQ)
        {
            passManager.AddWithName<DataflowPass>("AddRangeOfMarker").Configure(p =>
            {
                p.Add<Transform.Rules.Neutral.AddRangeOfAndMarker>();
                p.Add<Transform.Rules.Neutral.AddRangeOfAndMarkerOnFuncBody>();
            });
            passManager.AddWithName<EGraphPassWithQuantize>("AssignRanges");
        }
    }

    public async Task CompileAsync()
    {
        var target = _compileSession.Target;
        await RunPassAsync(p => TargetIndependentPass(p), "TargetIndependentPass");
        await RunPassAsync(p => target.RegisterTargetDependentPass(p, _compileSession.CompileOptions), "TargetDependentPass");

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
