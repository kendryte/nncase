// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.ArgsStruct;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

/// <summary>
/// Crop expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnCrop : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnCrop), 0, "input");

    /// <summary>
    /// Gets Args of Crop.
    /// </summary>
    public CropArgs Args { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return
            $"starts:{string.Join(",", (Args.Starts ?? Array.Empty<int>()).Select(x => x.ToString()))}, ends: {string.Join(",", (Args.Ends ?? Array.Empty<int>()).Select(x => x.ToString()))}, axes: {string.Join(",", (Args.Axes ?? Array.Empty<int>()).Select(x => x.ToString()))}";
    }
}
