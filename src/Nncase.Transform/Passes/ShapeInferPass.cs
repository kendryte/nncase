// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Transform.Rules;

namespace Nncase.Transform.Passes;

/// <summary>
/// Shape inference.
/// </summary>
public sealed class ShapeInferPass : DataflowPass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ShapeInferPass"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public ShapeInferPass(string name = "ShapeInfer")
        : base(name)
    {
        Add(new Rules.Neutral.IntegralPromotion());
        Add(new Rules.Neutral.FoldConstCall());
        Add(new Rules.Neutral.FoldShapeOf());
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction pre, RunPassOptions options)
    {
        BaseFunction post;
        int count = 0;
        RunPassOptions new_options = new(options);
        new_options.SetDumpDir(options.DumpDir);
        while (true)
        {
            post = (BaseFunction)CompilerServices.Rewrite(pre, Rules, new_options.SetPassName($"{Name}/Run_{count}"));
            if (post == pre)
            {
                break;
            }

            pre = post;
        }

        return Task.FromResult<BaseFunction>(post);
    }
}
