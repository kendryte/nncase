using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Studio.Services;
using Nncase.Studio.WinForms.Services;

namespace Microsoft.Extensions.DependencyInjection;

public static class NncaseStudioWinFormsDIExtensions
{
    public static IServiceCollection AddNncaseStudioWinForms(this IServiceCollection services)
    {
        services.AddSingleton<IFolderPicker, FolderPicker>();
        return services;
    }
}
