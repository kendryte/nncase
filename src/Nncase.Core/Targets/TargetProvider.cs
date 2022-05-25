// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Targets;

internal class TargetProvider : ITargetProvider
{
    private readonly Dictionary<string, ITarget> _targets;

    public TargetProvider(ITarget[] targets)
    {
        _targets = targets.ToDictionary(x => x.Kind);
    }

    public ITarget GetTarget(string name)
    {
        return _targets[name];
    }
}
