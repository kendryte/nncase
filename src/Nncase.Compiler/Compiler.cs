using Autofac;
using Autofac.Extensions.DependencyInjection;
using Autofac.Extras.CommonServiceLocator;
using CommonServiceLocator;
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
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Compiler;

public class Compiler
{
    private IRModule Module;
    public static void init(CompileOptions options)
    {
        // OrtKI.LoadDLL();
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
        loggingBuilder.AddConsole();
    }

    public IRModule ImportModule(Stream content, CompileOptions options)
    {
        CompilerServices.CompileOptions = options;
        Console.WriteLine($"Target: {options.Target}");
        var module = ImportModel(content, options);
        DumpModule(module, options, "ir_import");
        Console.WriteLine("Infer Shape...");
        DumpManager.RunWithDump("EvaluatorInShapeInfer", () => InferShape(module, options));
        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        DumpModule(module, options, "ir_infertype");
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        Console.WriteLine("ImportModule successful!");
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
        CompilerServices.CompileOptions = options;
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
        var dump_path = Path.Join(CompilerServices.CompileOptions.DumpDir, dirName);
        var pmgr = new PassManager(Module, new RunPassOptions(CompilerServices.GetCompileTarget, 3, Path.Join(CompilerServices.CompileOptions.DumpDir, dirName), CompilerServices.CompileOptions));
        register(pmgr);
        pmgr.RunAsync().Wait();
    }

    public void TargetIndependentPass(PassManager passManager, CompileOptions options)
    {
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

    public void Compile(CompileOptions options, QuantizeOptions quantOption)
    {
        options.QuantizeOptions = quantOption;
        CompilerServices.CompileOptions = options;
        var t = CompilerServices.GetCompileTarget;
        RunPass(p => TargetIndependentPass(p, options), "TargetIndependentPass");
        RunPass(p => t.RegisterTargetDependentPass(p, options), "TargetDependentPass");
        // RunPass(p => p.Add(new Quantization.EGraphPassWithBindQuantizeConfig("2.5_BindQuantizeConfig", options.QuantizeOptions!)));
        if (options.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            RunPass(p => t.RegisterQuantizePass(p, options), "QuantizePass");
            RunPass(p => t.RegisterTargetDependentAfterQuantPass(p, options), "TargetDependentAfterQuantPass");            
        }
        // fold constant
        RunPass(p => p.Add(new Transform.Passes.ShapeInferPass()), "ShapeInferAndFold");
        Console.WriteLine("Compile successful");
    }

    public void UsePTQ(QuantizeOptions quantOption)
    {
        CompilerServices.CompileOptions.QuantizeOptions = quantOption;
        CompilerServices.CompileOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
    }
    
    public byte[] Gencode()
    {
        var target = CompilerServices.GetCompileTarget;
        var moduleBuilder = new ModelBuilder(target, CompilerServices.CompileOptions);
        var linkedModel = moduleBuilder.Build(Module);
        using var output = new MemoryStream();
        linkedModel.Serialize(output);
        Console.WriteLine("Gencode successful");
        return output.ToArray();
    }
}