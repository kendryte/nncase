using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;

namespace NnCase.Cli
{
    public static class CliServiceCollectionExtensions
    {
        public static IServiceCollection AddCli(this IServiceCollection services)
        {
            return services
                .AddScoped<Compile>();
        }
    }
}
