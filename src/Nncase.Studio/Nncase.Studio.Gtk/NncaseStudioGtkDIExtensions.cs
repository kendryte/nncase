// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components;
using Nncase.Studio.Gtk.Blazor;
using Nncase.Studio.Photino.Services;
using Nncase.Studio.Services;

namespace Microsoft.Extensions.DependencyInjection;

public static class NncaseStudioGtkDIExtensions
{
    public static IServiceCollection AddNncaseStudioGtk(this IServiceCollection services)
    {
        services.AddSingleton<Dispatcher>(new GtkDispatcher(new GtkSynchronizationContext()));
        services.AddSingleton<IFolderPicker, FolderPicker>();
        return services;
    }
}
