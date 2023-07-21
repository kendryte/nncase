// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

[PatternFunctionalGenerator]
public sealed partial class CPUKernelOp : Op
{
    /// <summary>
    /// Gets the target.
    /// </summary>
    public Op Target { get; }

    public override string DisplayProperty() => Target.GetType().Name;
}
