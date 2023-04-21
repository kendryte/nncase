// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.Extensions.DependencyInjection;

public static class NncaseStudioDIExtensions
{
    public static IServiceCollection AddNncaseStudio(this IServiceCollection services)
    {
        services.AddAntDesign();

        return services;
    }
}
