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
    /// <summary>
    /// the main module
    /// </summary>
    public IRModule Module { get; private set; } = null!;

    private CompileOptions? _compileOptions = null;

    public CompileOptions CompileOptions => _compileOptions ?? throw new InvalidOperationException("Must UpdateCompileOptions First!");

    /// <summary>
    /// update compile options
    /// </summary>
    /// <param name="options"></param>
    public void UpdateCompileOptions(CompileOptions options)
    {
        _compileOptions = options;
    }

    public void Init()
    {
        var host = Host.CreateDefaultBuilder();
        host.ConfigureAppConfiguration(ConfigureAppConfiguration)
            .UseServiceProviderFactory(new AutofacServiceProviderFactory())
            .ConfigureContainer<ContainerBuilder>(ConfigureContainer)
            .ConfigureServices(ConfigureServices)
            .ConfigureLogging(ConfigureLogging)
            .UseConsoleLifetime();
        var iHost = host.Build();
        var provider = iHost.Services.GetRequiredService<ICompilerServicesProvider>();
        CompilerServices.Configure(provider);
    }

    private static void ConfigureContainer(ContainerBuilder builder)
    {
        var assemblies = ApplicationParts.LoadApplicationParts(c =>
        {
            c.AddCore()
                .AddEvaluator()
                .AddGraph()
                .AddEGraph()
                .AddStackVM();
        });
        builder.RegisterAssemblyModules(assemblies);
    }

    private static void ConfigureServices(HostBuilderContext context, IServiceCollection services)
    {
        services.AddLogging();
    }

    private static void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
    {
        builder.SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("config.json", true, false);
    }

    private static void ConfigureLogging(ILoggingBuilder loggingBuilder)
    {
        loggingBuilder.ClearProviders();
        loggingBuilder.AddConsole();
    }

    public IRModule ImportModule(Stream content)
    {
        //Console.WriteLine($"Target: {options.Target}");
        var module = ImportModel(content, CompileOptions);
        DumpModule(module, CompileOptions, "ir_import");
        //Console.WriteLine("Infer Shape...");

        if (CompileOptions.DumpLevel > 4)
            DumpManager.RunWithDump("EvaluatorInShapeInfer", () => InferShape(module, CompileOptions));
        else
            InferShape(module, CompileOptions);

        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        DumpModule(module, CompileOptions, "ir_infertype");
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        //Console.WriteLine("ImportModule successful!");
        return module;
    }

    private void InferShape(IRModule module, CompileOptions options)
    {
        var pmgr = new PassManager(module, new RunPassOptions(null!, options.DumpLevel, options.DumpDir));
        var constFold = new ShapeInferPass();
        pmgr.Add(constFold);
        pmgr.RunAsync().Wait();
    }

    private IRModule ImportModel(Stream content, CompileOptions options)
    {
        Module = options.InputFormat switch
        {
            "tflite" => Importers.ImportTFLite(content, options),
            "onnx" => Importers.ImportOnnx(content, options),
            _ => throw new NotImplementedException($"Not Implement {options.InputFormat} Impoter!"),
        };
        return Module;
    }

    private void DumpModule(IRModule module, CompileOptions options, string prefix)
    {
        var dumpPath = Path.Combine(options.DumpDir, "dump", prefix);
        CompilerServices.DumpIR(module.Entry!, prefix, dumpPath);
    }

    private void RunPass(Action<PassManager> register, string dirName)
    {
        var dump_path = Path.Join(CompileOptions.DumpDir, dirName);
        var pmgr = new PassManager(Module,
          new RunPassOptions(
              CompilerServices.GetTarget(CompileOptions.Target),
              CompileOptions.DumpLevel,
              Path.Join(CompileOptions.DumpDir, dirName),
              CompileOptions
          )
        );
        register(pmgr);
        pmgr.RunAsync().Wait();
    }

    public void TargetIndependentPass(PassManager passManager, CompileOptions options)
    {
        passManager.Add(new EGraphPass("1_NeutralOptimize"){
          new Transform.Rules.Neutral.FoldConstCall(),
          new Transform.Rules.Neutral.FoldNopTranspose(),
          new Transform.Rules.Neutral.FoldTwoTransposes(),
          new Transform.Rules.Neutral.CombineTransposeUnary(),
          new Transform.Rules.Neutral.CombineTransposePad(),
          new Transform.Rules.Neutral.CombineTransposeBinary(),
          new Transform.Rules.Neutral.CombineTransposeReduce(),
          new Transform.Rules.Neutral.CombineTransposeActivations(),
          new Transform.Rules.Neutral.FoldConv2DPads(),
        });
        if (options.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            AddMarker(passManager, options);
            AssignRange(passManager, options);
        }
    }

    public void AddMarker(PassManager passManager, CompileOptions options)
    {
        passManager.Add(new DataflowPass("add_rangeof_and_marker")
        {
            new Transform.Rules.Neutral.AddRangeOfAndMarkerToConv2D(),
            new Transform.Rules.Neutral.AddRangeOfAndMarkerToMatMul(),
        });
    }

    public void AssignRange(PassManager passManager, CompileOptions options)
    {
        passManager.Add(new Quantization.EGraphPassWithQuantize("1_AssignRanges", options.QuantizeOptions!));
    }

    public void Compile(CompileOptions? compileOptions = null)
    {
        var t = CompilerServices.GetTarget(CompileOptions.Target);
        if (CompileOptions.DumpLevel > 4)
            DumpManager.RunWithDump("TargetIndependentEval", () => RunPass(p => TargetIndependentPass(p, CompileOptions), "TargetIndependentPass"));
        else
            RunPass(p => TargetIndependentPass(p, CompileOptions), "TargetIndependentPass");
        RunPass(p => t.RegisterTargetDependentPass(p, CompileOptions), "TargetDependentPass");
        // RunPass(p => p.Add(new Quantization.EGraphPassWithBindQuantizeConfig("2.5_BindQuantizeConfig", CompileOptions.QuantizeOptions!)));
        if (CompileOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            RunPass(p => t.RegisterQuantizePass(p, CompileOptions), "QuantizePass");
            RunPass(p => t.RegisterTargetDependentAfterQuantPass(p, CompileOptions), "TargetDependentAfterQuantPass");
            var clear = new DataflowPass("ClearMarker") { new RemoveMarker() };
            RunPass(p => p.Add(clear), "RemoveMarker");
        }

        // fold constant
        RunPass(p => p.Add(new Transform.Passes.ShapeInferPass()), "ShapeInferAndFold");
        // Console.WriteLine("Compile successful");
    }

    // /// <summary>
    // /// this interface for python
    // /// </summary>
    // /// <param name="quantOption"></param>
    // public void UsePTQ(QuantizeOptions quantOption)
    // {
    //     CompilerServices.CompileOptions.QuantizeOptions = quantOption;
    //     CompilerServices.CompileOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
    // }

    public byte[] Gencode()
    {
        var target = CompilerServices.GetTarget(CompileOptions.Target);
        var moduleBuilder = new ModelBuilder(target, CompileOptions);
        var linkedModel = moduleBuilder.Build(Module);
        using var output = new MemoryStream();
        linkedModel.Serialize(output);
        return output.ToArray();
    }
}
