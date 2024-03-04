// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Schedule;
using Xunit;
using F = Nncase.IR.F;

namespace Nncase.Tests.ScheduleTest;

public class UnitTestTilingSolver
{
    [Fact]
    public void TestSimpleFor()
    {
        var schedule = Tiling.AutoTile();
        Debug.WriteLine(schedule);
    }
}
