using System.IO;
using System.Runtime.CompilerServices;
using Autofac;
using Autofac.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.Evaluator;
using Nncase.Evaluator.Ops;
using Nncase.IR;
using Nncase.Transform;
using Xunit;

namespace Nncase.Tests
{
    public class TestingConfiguration
    {
        public string LogDir { get; set; }
    }
    
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
            builder.RegisterModule<EvaluatorModule>();
        }
        
        private void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
        {
            builder.SetBasePath(Path.GetDirectoryName(Testing.GetTestingFilePath()))
                          .AddJsonFile("config.json", true, false);
            Configuration = builder.Build();
        }

        private void ConfigureServices(HostBuilderContext context, IServiceCollection services)
        {
            services.Configure<TestingConfiguration>(options => Configuration.GetSection("Testing").Bind(options));
        }
    }
}