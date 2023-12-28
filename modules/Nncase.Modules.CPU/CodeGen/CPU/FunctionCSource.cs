// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#define MULTI_CORE_XPU
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.CodeGen;

internal sealed record KernelCSource(string Main, string Kernel)
{
}
