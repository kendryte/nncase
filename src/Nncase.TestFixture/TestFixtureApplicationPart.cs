// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// EGraph application part extensions.
/// </summary>
public static class TestFixtureApplicationPart
{
    /// <summary>
    /// Add egraph assembly.
    /// </summary>
    /// <param name="assemblies">Assembly collection.</param>
    /// <returns>Updated assembly collection.</returns>
    public static IList<Assembly> AddTestFixture(this IList<Assembly> assemblies)
    {
        assemblies.Add(typeof(TestFixture.UnitTestFixtrue).Assembly);
        return assemblies;
    }
}
