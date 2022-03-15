// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Model.
/// </summary>
public sealed class IRModel
{

    /// <summary>
    /// contains modules.
    /// </summary>
    public readonly List<IRModule> Modules;

    /// <summary>
    /// the entry
    /// </summary>
    public IR.Callable Entry;

    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="sub_modules"></param>
    public IRModel(IEnumerable<IRModule> sub_modules)
    {
        Modules = new(sub_modules);
        var entrys = (from m in Modules
                      where m.Entry is not null
                      select m.Entry).ToArray();
        if (entrys.Length is var l && (l == 0 || l > 1))
            throw new InvalidOperationException("Invalid Entry!");
        Entry = entrys[0];
    }
}
