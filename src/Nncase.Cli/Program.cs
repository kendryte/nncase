// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine.Builder;
using System.CommandLine.Hosting;
using System.CommandLine.Parsing;
using System.IO;
using System.Threading.Tasks;
using Autofac;
using Autofac.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase.Hosting;

namespace Nncase.Cli
{
    internal partial class Program
    {
        public static async Task<int> Main(string[] args)
        {
            return await BuildCommandLine()
                .UseHost(
                    _ => Host.CreateDefaultBuilder(args),
                    host =>
                    {
                        host.ConfigureAppConfiguration(ConfigureAppConfiguration)
                            .UseServiceProviderFactory(new AutofacServiceProviderFactory())
                            .ConfigureContainer<ContainerBuilder>(ConfigureContainer)
                            .ConfigureServices(ConfigureServices)
                            .ConfigureLogging(ConfigureLogging)
                            .UseConsoleLifetime();
                    })
                .UseDefaults()
                .Build().InvokeAsync(args);
        }

        private static void ConfigureContainer(ContainerBuilder builder)
        {
            var assemblies = ApplicationParts.LoadApplicationParts(c =>
            {
                c.AddCore()
                .AddEvaluator()
                .AddGraph()
                .AddEGraph();
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
    }
}
