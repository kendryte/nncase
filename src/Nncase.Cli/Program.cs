// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CommandLine.Builder;
using System.CommandLine.Hosting;
using System.CommandLine.Parsing;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Nncase.Hosting;

namespace Nncase.Cli;

internal partial class Program
{
    public static async Task<int> Main(string[] args)
    {
        return await BuildCommandLine()
            .UseHost(ConfigureHost)
            .UseDefaults()
            .Build().InvokeAsync(args);
    }

    private static void ConfigureHost(IHostBuilder hostBuilder)
    {
        hostBuilder.ConfigureAppConfiguration(ConfigureAppConfiguration)
            .UseConsoleLifetime()
            .ConfigureCompiler();
    }

    private static void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
    {
        var baseDirectory = Path.GetDirectoryName(typeof(Program).Assembly.Location);
        builder.SetBasePath(baseDirectory)
            .AddJsonFile("config.json", true, false);
    }
}
