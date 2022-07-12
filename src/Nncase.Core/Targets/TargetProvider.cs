// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;

namespace Nncase.Targets;

internal class TargetProvider : ITargetProvider
{
    private readonly Dictionary<string, ITarget> _targets;

    public TargetProvider(ITarget[] targets, IConfiguration configure)
    {
        _targets = targets.ToDictionary(x => x.Kind);
        foreach (var target in _targets.Values)
        {
            target.ParseTargetDependentOptions(configure.GetSection(target.Kind));
        }
    }

    public ITarget GetTarget(string name)
    {
        return _targets[name];
    }
}
