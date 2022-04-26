// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime;

internal sealed class RTModel : IRTModel
{
    public RTModel(IReadOnlyList<IRTModule> modules, IRTFunction entry)
    {
        Modules = modules;
        Entry = entry;
    }

    /// <inheritdoc/>
    public IReadOnlyList<IRTModule> Modules { get; }

    /// <inheritdoc/>
    public IRTFunction Entry { get; }

    /// <inheritdoc/>
    public async ValueTask InitializeAsync()
    {
        foreach (var module in Modules)
        {
            await module.InitializeAsync();
        }
    }

    /// <inheritdoc/>
    public async ValueTask UninitializeAsync()
    {
        foreach (var module in Modules)
        {
            await module.UninitializeAsync();
        }
    }
}
