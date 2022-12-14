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
                    _ => CompilerHost.CreateHostBuilder(args),
                    host =>
                    {
                        host.UseConsoleLifetime();
                    })
                .UseDefaults()
                .Build().InvokeAsync(args);
        }
    }
}
