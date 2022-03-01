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
        Add(new Rules.Neutral.FoldConstCall());
        Add(new Rules.Neutral.FoldShapeOf());
    }

    /// <inheritdoc/>
    protected override Function RunCore(Function pre, RunPassOptions options)
    {
        Function post;
        int count = 0;
        RunPassOptions new_options = new(options);
        new_options.SetDir(options.PassDumpDir);
        while (true)
        {
            post = (Function)CompilerServices.Rewrite(pre, Rules, new_options.SetName($"{Name}/Run_{count}"));
            if (post == pre)
            {
                if (!CompilerServices.InferenceType(post))
                {
                    throw new InvalidOperationException("Can't InferShape For This Model!");
                }
                else
                {
                    break;
                }
            }

            pre = post;
        }

        return post;
    }
}
