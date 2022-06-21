// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.K210;

/// <summary>
/// Fake KPU Upload.
/// </summary>
[PatternFunctionalGenerator]
public sealed record class FakeKPUUpload : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(FakeKPUDownload), 0, "input", HasRank(4));
}
