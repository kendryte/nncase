using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Nncase.Studio.WinForms;

internal static class Program
{
    /// <summary>
    ///  The main entry point for the application.
    /// </summary>
    [STAThread]
    public static void Main()
    {
        var host = Host.CreateDefaultBuilder()
         .ConfigureServices(ConfigureServices)
         .ConfigureLogging(ConfigureLogging)
         .Build();

        var services = host.Services;
        var mainForm = services.GetRequiredService<MainForm>();
        Application.Run(mainForm);
    }

    private static void ConfigureLogging(HostBuilderContext context, ILoggingBuilder loggingBuilder)
    {
        loggingBuilder.ClearProviders();
        loggingBuilder.AddConsole();
    }

    private static void ConfigureServices(HostBuilderContext context, IServiceCollection services)
    {
        services.AddWindowsFormsBlazorWebView();
        services.AddNncaseStudio();
        services.AddNncaseStudioWinForms();
#if DEBUG
        services.AddBlazorWebViewDeveloperTools();
#endif

        services.AddTransient<MainForm>();
    }
}
