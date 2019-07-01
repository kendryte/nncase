using System;
using System.IO;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace NnCase.Cli
{
    public class CommandArgsOptions : IOptions<CommandArgsOptions>
    {
        CommandArgsOptions IOptions<CommandArgsOptions>.Value => this;

        public string[] Args { get; set; }
    }

    public class Program
    {
        public static async Task Main(string[] args)
        {
            AppDomain.CurrentDomain.UnhandledException += (s, e) =>
            {
                if (e.ExceptionObject is Exception ex)
                {
                    Console.WriteLine("Fatal: " + ex.Message);

                    Console.WriteLine(ex.ToString());
                }
                else
                {
                    Console.WriteLine("Fatal: Unexpected error occurred.");
                }

                Environment.Exit(-1);
            };

            var services = new ServiceCollection();
            ConfigureServices(services);
            services.Configure<CommandArgsOptions>(o => o.Args = args);

            var serviceProvider = services.BuildServiceProvider();
            var iface = ActivatorUtilities.CreateInstance<Interface>(serviceProvider);
            await iface.RunAsync(default);
        }

        private static void ConfigureServices(IServiceCollection services)
        {
            services
                .AddOptions()
                .AddLogging(ConfigureLogging)
                .AddCli();
        }

        private static void ConfigureLogging(ILoggingBuilder loggingBuilder)
        {
            loggingBuilder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information);
        }
    }
}
