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
/// ConvTranspose expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnConvTranspose : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnConvTranspose), 0, "input");

    public ConvTransposeArgs Args { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return Args.ToString();
    }
}
