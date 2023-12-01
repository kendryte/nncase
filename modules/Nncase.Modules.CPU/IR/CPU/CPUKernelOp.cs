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

public sealed class CPUKernelOp : Op
{
    private readonly ExprPinner _exprPinner;

    public CPUKernelOp(Op target)
    {
        _exprPinner = new(target);
        Target = target;
    }

    /// <summary>
    /// Gets the target.
    /// </summary>
    public Op Target { get; }

    /// <inheritdoc/>
    public override IEnumerable<ParameterInfo> Parameters => Target.Parameters;

    public override string DisplayProperty() => Target.GetType().Name;
}
