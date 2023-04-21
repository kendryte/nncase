// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Drawing;
using System.IO;
using System.Text;
using Gtk;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.Studio.Gtk.Blazor;

namespace Nncase.Studio.Gtk;

internal class Program
{
    private static Application _app = default!;

    [STAThread]
    public static void Main(string[] args)
    {
        var builder = Host.CreateDefaultBuilder();
        ConfigureHost(builder);
        var host = builder.Build();

        Application.Init();
        _app = new Application("Nncase Studio", GLib.ApplicationFlags.None);
        _app.Register(GLib.Cancellable.Current);

        var window = new Window(WindowType.Toplevel);
        window.DefaultSize = new Gdk.Size(1024, 768);

        window.DeleteEvent += (o, e) =>
        {
            Application.Quit();
        };

        var webView = new BlazorWebView(host.Services);
        webView.RootComponents.Add<Main>("#app");
        window.Add(webView);

#if DEBUG
        webView.Settings.EnableDeveloperExtras = true;
#endif

        window.ShowAll();
        Application.Run();
    }

    private static void ConfigureHost(IHostBuilder hostBuilder)
    {
        hostBuilder.ConfigureAppConfiguration(ConfigureAppConfiguration)
            .ConfigureServices(ConfigureServices);
    }

    private static void ConfigureServices(HostBuilderContext context, IServiceCollection services)
    {
        services.AddNncaseStudio();
        services.AddNncaseStudioGtk();
        services.AddBlazorWebViewOptions(options =>
        {
            options.HostPath = "wwwroot/index.html";
        });
    }

    private static void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
    {
        var baseDirectory = Path.GetDirectoryName(typeof(Program).Assembly.Location)!;
        builder.SetBasePath(baseDirectory)
            .AddJsonFile("config.json", true, false);
    }
}
