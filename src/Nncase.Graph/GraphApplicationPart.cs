// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Nncase.Hosting;
using Nncase.Passes;
using Nncase.PatternMatch;

namespace Nncase;

/// <summary>
/// Graph application part extensions.
/// </summary>
public static class GraphApplicationPart
{
    /// <summary>
    /// Add graph assembly.
    /// </summary>
    /// <param name="registrator">Service registrator.</param>
    /// <returns>Configured service registrator.</returns>
    public static IRegistrator AddGraph(this IRegistrator registrator)
    {
        return registrator.RegisterModule<PatternMatchModule>()
            .RegisterModule<TransformModule>();
    }
}
