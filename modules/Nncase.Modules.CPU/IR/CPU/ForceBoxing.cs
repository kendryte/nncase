// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

/// <summary>
/// Force Boxing, only can change broadcast to partial, just use for test ccl.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ForceBoxing : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ForceBoxing), 0, "input", ParameterKind.Input);

    public DistributedType NewType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"NewType: {NewType}";
}
