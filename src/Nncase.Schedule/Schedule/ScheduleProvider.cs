// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule;

internal class ScheduleProvider : IScheduleProvider
{
    public Call Tile(Grid grid, IRModule module)
    {
        var scheduler = new AffineTiler(grid);
        return scheduler.Tile(module);
    }
}
