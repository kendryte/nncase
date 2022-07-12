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
using Nncase.Transform;
using Nncase.Transform.Passes;
using OrtKISharp;

namespace Nncase.Compiler;

public class Compiler
{
    private IRModule Module;
    private CompileOptions Options;
    public static void init(CompileOptions options)
    {
        OrtKI.LoadDLL();
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
        Options = options;
        Console.WriteLine($"Target: {options.Target}");
        var module = ImportModel(content, options);
        Console.WriteLine("Infer Shape...");
        InferShape(module, options);
        var inferSucc = CompilerServices.InferenceType(module.Entry!);
        DumpModule(module, options, "ir_import");
        if (!inferSucc)
        {
            throw new InvalidOperationException("InferShape Failed For This Model!");
        }

        DumpManager.RunWithDump("EvaluatorInShapeInfer", () => InferShape(module, options));
        
        DumpModule(module, options, "ir_infertype");
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
        Module = options.InputFormat switch
        {
            "tflite" => Importers.ImportTFLite(content),
            "onnx" => Importers.ImportOnnx(content),
            _ => throw new NotImplementedException($"Not Implement {options.InputFormat} Impoter!"),
        };
        return Module;
    }


    private void DumpModule(IRModule module, CompileOptions options, string prefix)
    {
        var dumpPath = Path.Combine(options.DumpDir, "dump", prefix);
        CompilerServices.DumpIR(module.Entry!, prefix, dumpPath);
    }

    private void RunPass(Action<PassManager> register)
    {
        // todo:dump dir
        var pmgr = new PassManager(Module, new RunPassOptions(CompilerServices.GetTarget(Options.Target), 0, "null", Options));
        register(pmgr);
        pmgr.RunAsync().Wait();
    }

    public void TargetIndependentPass()
    {
    }
    
    public void Compile(CompileOptions options)
    {
        Options = options;
        var t = CompilerServices.GetTarget(options.Target);
        TargetIndependentPass();
        RunPass(p => t.RegisterTargetDependentPass(p, options));
        RunPass(p => t.RegisterTargetDependentAfterQuantPass(p));
        Console.WriteLine("Compile successful");
    }

    public byte[] Gencode()
    {
        var target = CompilerServices.GetTarget(Options.Target);
        var moduleBuilder = new ModelBuilder(target);
        var linkedModel = moduleBuilder.Build(Module);
        using var output = new MemoryStream();
        linkedModel.Serialize(output);
        Console.WriteLine("Gencode successful");
        return output.ToArray();
    }
}