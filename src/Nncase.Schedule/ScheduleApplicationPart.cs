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

namespace Nncase;

/// <summary>
/// Schedule application part extensions.
/// </summary>
public static class ScheduleApplicationPart
{
    /// <summary>
    /// Add schedule assembly.
    /// </summary>
    /// <param name="registrator">Service registrator.</param>
    /// <returns>Configured service registrator.</returns>
    public static IRegistrator AddSchedule(this IRegistrator registrator)
    {
        return registrator.RegisterModule<ScheduleModule>();
    }
}
