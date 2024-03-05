// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;
using Nncase.Schedule;

namespace Nncase.Passes;

/// <summary>
/// Schedule module.
/// </summary>
internal class ScheduleModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<ScheduleProvider>(reuse: Reuse.Singleton);
    }
}
