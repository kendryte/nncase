// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Shape inference.
/// </summary>
public sealed class ShapeInferPass : DataflowPass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ShapeInferPass"/> class.
    /// </summary>
    public ShapeInferPass()
    {
        Add<Rules.Neutral.IntegralPromotion>();
        Add<Rules.Neutral.FoldConstCall>();
        Add<Rules.Neutral.FoldShapeOf>();
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction pre, RunPassContext options)
    {
        BaseFunction post;
        int count = 0;

        while (true)
        {
            using var dumpScope = new DumpScope($"Run_{count}");
            post = (BaseFunction)CompilerServices.Rewrite(pre, Rules, options);
            if (post == pre)
            {
                break;
            }

            pre = post;
        }

        return Task.FromResult(post);
    }
}
