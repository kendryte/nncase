using Autofac;
using Autofac.Extensions.DependencyInjection;
using Autofac.Extras.CommonServiceLocator;
using CommonServiceLocator;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.Transform;
using OrtKISharp;

namespace Nncase.Compiler;

public class Compiler
{
    public static void init()
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
                .AddEvaluator();
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
        Console.WriteLine($"Target: {options.Target}");
        var module = ImportModel(content, options);
        DumpModule(module, options, "ir_import");
        if (!CompilerServices.InferenceType(module.Entry))
        {
            InferShape(module, options);
        }

        DumpModule(module, options, "ir_infertype");
        Console.WriteLine("ImportModule successful!");
        return module;
    }

    private void InferShape(IRModule module, CompileOptions options)
    {
        Console.WriteLine("Infer Shape...");
        var pmgr = new PassManager(module, new RunPassOptions(null, options.DumpLevel, options.DumpDir));
        var constFold = new ShapeInferPass();
        pmgr.Add(constFold);
        pmgr.Run();
    }

    private IRModule ImportModel(Stream content, CompileOptions options) =>
        options.InputFormat switch
        {
            "tflite" => Importers.ImportTFLite(content),
            "onnx" => Importers.ImportOnnx(content),
            _ => throw new NotImplementedException($"Not Implement {options.InputFormat} Impoter!"),
        };

    private void DumpModule(IRModule module, CompileOptions options, string prefix)
    {
        var dumpPath = Path.Combine(options.DumpDir, "dump", prefix);
        Directory.CreateDirectory(dumpPath);

        var func = module.Entry;
        using var dumpFile = File.Open(Path.Combine(dumpPath, $"{func.Name}.il"), FileMode.OpenOrCreate);
        using var dumpWriter = new StreamWriter(dumpFile);
        IRPrinter.DumpFunctionAsIL(func, dumpWriter);
    }
}