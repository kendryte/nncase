// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Tomlyn.Extensions.Configuration;

namespace Nncase.Tests;

public class Startup
{
    public void ConfigureHost(IHostBuilder hostBuilder) =>
        hostBuilder
            .ConfigureAppConfiguration(ConfigureAppConfiguration)
            .ConfigureCompiler(builder =>
            {
                builder.ConfigureModules(c =>
                {
                    c.AddTests();
                    c.AddTestFixture();
                });
            });

    public void Configure(IHost host)
    {
        CompilerServices.Configure(host.Services);
    }

    private void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
    {
        builder.Sources.Clear(); // CreateDefaultBuilder adds default configuration sources like appsettings.json. Here we can remove them
        builder.AddTomlFile("testsettings.toml", true, false);
    }
}
