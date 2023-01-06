// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// Options for running pass.
/// </summary>
public sealed record RunPassContext
{
    /// <summary>
    /// Gets or sets pass index in a <see cref="PassManager"/>.
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether control rewrite once or not.
    /// Default is false.
    /// </summary>
    public bool RewriteOnce { get; set; }

    /// <summary>
    /// Gets or sets the match option.
    /// </summary>
    public MatchOptions MatchOptions { get; set; } = new MatchOptions();
}
