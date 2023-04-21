// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Studio.Gtk.Blazor;

namespace Microsoft.Extensions.DependencyInjection;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddBlazorWebViewOptions(this IServiceCollection services, Action<BlazorWebViewOptions> configureOptions)
    {
        return services
            .AddBlazorWebView()
            .Configure<BlazorWebViewOptions>(configureOptions);
    }
}
