// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Maui;
using Nncase.Studio.Maui.Services;
using Nncase.Studio.Services;

namespace Microsoft.Extensions.DependencyInjection;

public static class NncaseStudioMauiDIExtensions
{
    public static IServiceCollection AddNncaseStudioMaui(this IServiceCollection services)
    {
        services.AddSingleton<IFolderPicker, FolderPicker>();
        return services;
    }
}
