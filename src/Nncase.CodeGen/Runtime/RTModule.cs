// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime;

public abstract class RTModule : IRTModule
{
    public RTModule(IReadOnlyList<IRTFunction> functions)
    {
        Functions = functions;
    }

    public IReadOnlyList<IRTFunction> Functions { get; }

    public virtual async ValueTask InitializeAsync()
    {
        foreach (var function in Functions)
        {
            await function.InitializeAsync();
        }
    }

    public virtual async ValueTask UninitializeAsync()
    {
        foreach (var function in Functions)
        {
            await function.UninitializeAsync();
        }
    }
}
