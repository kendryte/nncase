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

            var hostBuilder = new HostBuilder()
                .UseContentRoot(Path.GetDirectoryName(typeof(Program).Assembly.Location))
                .ConfigureLogging(ConfigureLogging)
                .ConfigureServices(ConfigureServices)
                .ConfigureServices(c =>
                {
                    c.Configure<CommandArgsOptions>(x => x.Args = args);
                });

            await hostBuilder.RunConsoleAsync();
        }

        private static void ConfigureServices(HostBuilderContext context, IServiceCollection services)
        {
            services
                .AddOptions()
                .AddLogging()
                .AddCli()
                .AddHostedService<Interface>();
        }

        private static void ConfigureLogging(HostBuilderContext context, ILoggingBuilder loggingBuilder)
        {
            loggingBuilder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information);
        }
    }
}
