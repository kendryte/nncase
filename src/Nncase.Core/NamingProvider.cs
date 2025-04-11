// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Nncase;

public interface INamingProvider
{
    string GetName(string prefix);
}

internal class NamingProvider : INamingProvider
{
    private readonly Dictionary<string, int> _nameCount = new();

    public string GetName(string prefix)
    {
        if (_nameCount.TryGetValue(prefix, out var count))
        {
            _nameCount[prefix] = count + 1;
            return $"{prefix}_{count}";
        }
        else
        {
            _nameCount[prefix] = 1;
            return prefix;
        }
    }
}
