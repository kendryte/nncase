// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.CPU;

internal class CPURTModule
{
    /// <summary>
    /// KPU module kind.
    /// </summary>
    public static readonly string Kind = "cpu";

    /// <summary>
    /// KPU module version.
    /// </summary>
    public static readonly uint Version = 1;
}
