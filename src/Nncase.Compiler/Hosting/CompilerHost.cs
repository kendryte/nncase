// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using DryIoc.Microsoft.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Nncase.Hosting;

/// <summary>
/// Compiler host helper.
/// </summary>
public static class CompilerHost
{
    /// <summary>
    /// Create compiler host builder.
    /// </summary>
    /// <param name="args">Commandline arguments.</param>
    /// <param name="configureHost">Configure host builder.</param>
    /// <returns>Created host builder.</returns>
    public static IHostBuilder CreateHostBuilder(string[]? args = null, Action<IHostBuilder>? configureHost = null)
    {
        var host = Host.CreateDefaultBuilder(args);
        host.UseServiceProviderFactory(new DryIocServiceProviderFactory());

        configureHost?.Invoke(host);
        host.ConfigureContainer<Container>(ConfigureBuiltinModules)
            .ConfigureServices(ConfigureServices)
            .ConfigureLogging(ConfigureLogging)
            .ConfigureContainer<Container>(ConfigurePlugins);
        return host;
    }

    private static void ConfigureBuiltinModules(Container builder)
    {
        builder.AddCore()
                .AddEvaluator()
                .AddGraph()
                .AddEGraph()
                .AddStackVM();
    }

    private static void ConfigureServices(HostBuilderContext context, IServiceCollection services)
    {
        services.AddLogging();

        services.AddSingleton<PluginLoader>();
        services.AddScoped<ICompiler, Compiler.Compiler>();
    }

    private static void ConfigureLogging(ILoggingBuilder loggingBuilder)
    {
        loggingBuilder.ClearProviders();
        loggingBuilder.AddConsole();
    }

    private static void ConfigurePlugins(Container builder)
    {
        var pluginLoader = builder.Resolve<PluginLoader>();
        var plugins = pluginLoader.LoadPlugins();
        foreach (var plugin in plugins)
        {
            plugin.ConfigureServices(builder);
        }
    }
}
