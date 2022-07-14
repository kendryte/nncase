// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// custom call op.
/// </summary>
public sealed record CustomCall(CustomOp CustomOp) : Op
{

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return CustomOp.RegisteredName;
    }
}
