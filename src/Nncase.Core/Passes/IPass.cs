// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;

namespace Nncase.Passes;

/// <summary>
/// IR or TIR pass.
/// </summary>
public interface IPass
{
    /// <summary>
    /// Gets or sets pass name.
    /// </summary>
    string Name { get; set; }

    /// <summary>
    /// Gets required collection of <see cref="IAnalysisResult"/>.
    /// </summary>
    IReadOnlyCollection<Type> AnalysisTypes { get; }
}
