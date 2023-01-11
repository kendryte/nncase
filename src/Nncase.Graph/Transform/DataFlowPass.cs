// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// Dataflow pass.
/// </summary>
public class DataflowPass : RulesPass
{
    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassContext options)
    {
        return Task.FromResult((BaseFunction)CompilerServices.Rewrite(function, Rules, options));
    }
}
