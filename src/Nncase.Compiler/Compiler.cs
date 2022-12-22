using Autofac;
using Autofac.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.CodeGen;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Nncase.Transform.Rules.Lower;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Compiler;

public class Compiler
{
    private readonly CompileOptions _compileOptions;
    private IRModule? _module;

    public Compiler(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
    }

    public static void Initialize()
    {
        var iHost = CompilerHost.CreateHostBuilder().Build();
        var provider = iHost.Services.GetRequiredService<ICompilerServicesProvider>();
        CompilerServices.Configure(provider);
    }

    public IRModule ImportModule(Stream content)
    {
        CompilerServices.CompileOptions = _compileOptions;
        //Console.WriteLine($"Target: {options.Target}");
        var module = ImportModel(content);
        DumpModule(module, "ir_import");
        //Console.WriteLine("Infer Shape...");

        if (_compileOptions.DumpLevel > 4)
            DumpManager.RunWithDump("EvaluatorInShapeInfer", () => RunPass(pmg => pmg.Add(new ShapeInferPass()), "ShapeInferAfterImport"));
        else
            RunPass(pmg => pmg.Add(new ShapeInferPass()), "ShapeInferAfterImport");

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        DumpModule(module, "ir_infertype");
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        //Console.WriteLine("ImportModule successful!");
        return module;
    }

    private IRModule ImportModel(Stream content)
    {
        _module = _compileOptions.InputFormat switch
        {
            "tflite" => Importers.ImportTFLite(content, _compileOptions),
            "onnx" => Importers.ImportOnnx(content, _compileOptions),
            _ => throw new NotImplementedException($"Not Implement {_compileOptions.InputFormat} Impoter!"),
        };
        return _module;
    }

    private void DumpModule(IRModule module, string prefix)
    {
        var dumpPath = Path.Combine(_compileOptions.DumpDir, "dump", prefix);
        CompilerServices.DumpIR(module.Entry!, prefix, dumpPath);
    }

    private void RunPass(Action<PassManager> register, string dirName)
    {
        var pmgr = new PassManager(_module, new RunPassOptions(CompilerServices.GetTarget(_compileOptions.Target), _compileOptions.DumpLevel, Path.Join(_compileOptions.DumpDir, dirName), _compileOptions));
        register(pmgr);
        pmgr.RunAsync().Wait();
    }

    public void TargetIndependentPass(PassManager passManager)
    {
        if (_compileOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
            passManager.Add(new EGraphPass("1_NeutralOptimize"){
          new Transform.Rules.Neutral.FoldConstCall(),
          new Transform.Rules.Neutral.FoldNopTranspose(),
          new Transform.Rules.Neutral.FoldTwoTransposes(),
          new Transform.Rules.Neutral.CombineTransposeUnary(),
          new Transform.Rules.Neutral.CombineTransposePad(),
          new Transform.Rules.Neutral.CombineTransposeBinary(),
          new Transform.Rules.Neutral.CombineTransposeReduce(),
          new Transform.Rules.Neutral.CombineTransposeActivations(),
          new Transform.Rules.Neutral.CombinePadTranspose(),
          new Transform.Rules.Neutral.FoldNopPad(),
          new Transform.Rules.Neutral.FoldConv2DPads(),
          new Transform.Rules.Neutral.FoldReduceWindow2DPads(),
        });
        if (_compileOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            passManager.Add(new DataflowPass("2_AddRangeOfMarker")
            {
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2D(),
                new Transform.Rules.Neutral.AddRangeOfAndMarkerToMatMul(),
                // new Transform.Rules.Neutral.AddRangeOfAndMarkerToRedeceWindow2D(),
                // new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2DTranspose(),
                // new Transform.Rules.Neutral.AddRangeOfAndMarkerToBinary(),
            });
            passManager.Add(new Quantization.EGraphPassWithQuantize("3_AssignRanges", _compileOptions.QuantizeOptions!));
        }
    }

    public void Compile()
    {
        var t = CompilerServices.GetTarget(_compileOptions.Target);
        if (_compileOptions.DumpLevel > 4)
            DumpManager.RunWithDump("TargetIndependentEval", () => RunPass(p => TargetIndependentPass(p), "TargetIndependentPass"));
        else
            RunPass(p => TargetIndependentPass(p), "TargetIndependentPass");
        RunPass(p => t.RegisterTargetDependentPass(p, _compileOptions), "TargetDependentPass");
        // RunPass(p => p.Add(new Quantization.EGraphPassWithBindQuantizeConfig("2.5_BindQuantizeConfig", options.QuantizeOptions!)));
        if (_compileOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            RunPass(p => t.RegisterQuantizePass(p, _compileOptions), "QuantizePass");
            RunPass(p => t.RegisterTargetDependentAfterQuantPass(p, _compileOptions), "TargetDependentAfterQuantPass");
            RunPass(t => t.Add(new DataflowPass("ClearMarker") { new RemoveMarker() }), "RemoveMarker");
        }

        // fold constant
        RunPass(p => p.Add(new Transform.Passes.ShapeInferPass()), "ShapeInferAfterCompile");
        // Console.WriteLine("Compile successful");
    }

    public void Gencode(Stream output)
    {
        var target = CompilerServices.GetTarget(_compileOptions.Target);
        var moduleBuilder = new ModelBuilder(target, _compileOptions);
        var linkedModel = moduleBuilder.Build(_module);
        linkedModel.Serialize(output);
        // Console.WriteLine("Gencode successful");
    }
}
