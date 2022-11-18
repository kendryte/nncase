using System;
using System.IO;
using System.Runtime.CompilerServices;
using Autofac;
using Autofac.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.Evaluator;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.TestFixture;
using Nncase.Transform;
using Tomlyn.Extensions.Configuration;
using Xunit;

namespace Nncase.Tests;

public class Startup
{
    public IConfigurationRoot Configuration { get; set; }

    public void ConfigureHost(IHostBuilder hostBuilder) =>
        hostBuilder
            .ConfigureAppConfiguration(ConfigureAppConfiguration)
            .UseServiceProviderFactory(new AutofacServiceProviderFactory())
            .ConfigureContainer<ContainerBuilder>(ConfigureContainer)
            .ConfigureServices(ConfigureServices);

    private static void ConfigureContainer(ContainerBuilder builder)
    {
        var assemblies = ApplicationParts.LoadApplicationParts(c =>
        {
            c.AddCore()
            .AddEvaluator()
            .AddGraph()
            .AddEGraph()
            .AddStackVM()
            .AddK210()
            .AddTestFixture();
        });
        builder.RegisterAssemblyModules(assemblies);
    }

    private void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
    {
        builder.Sources.Clear(); // CreateDefaultBuilder adds default configuration sources like appsettings.json. Here we can remove them
        builder.AddTomlFile("testsettings.toml", true, false);
        Configuration = builder.Build();
    }

    private void ConfigureServices(HostBuilderContext context, IServiceCollection services)
    {
        services.Configure<CompileOptions>(options =>
        {
            Configuration.GetSection("CompileOptions").Bind(options);
            options.QuantType = Configuration["CompileOptions:QuantType"] switch
            {
                "Int8" => DataTypes.Int8,
                "UInt8" => DataTypes.UInt8,
                _ => throw new System.ArgumentOutOfRangeException(),
            };
        });
    }

    public void Configure(ICompilerServicesProvider provider, ITestingProvider testing_provider)
    {
        Environment.SetEnvironmentVariable("NNCASE_TARGET_PATH", "");

        CompilerServices.Configure(provider);
        TestFixture.Testing.Configure(testing_provider);
        if (CompilerServices.CompileOptions.DumpDir == "")
            CompilerServices.CompileOptions.DumpDir = TestFixture.Testing.GetDumpDirPath();
    }
}
