// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// BatchToSpace expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class BatchToSpace : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BatchToSpace), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets block shape.
    /// </summary>
    public static readonly ParameterInfo BlockShape = new(typeof(BatchToSpace), 1, "blockShape");

    /// <summary>
    /// Gets crops.
    /// </summary>
    public static readonly ParameterInfo Crops = new(typeof(BatchToSpace), 2, "crops");
}
