using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Core application part extensions.
/// </summary>
public static class CoreApplicationPart
{
    /// <summary>
    /// Add core assembly.
    /// </summary>
    /// <param name="assemblies">Assembly collection.</param>
    /// <returns>Updated assembly collection.</returns>
    public static IList<Assembly> AddCore(this IList<Assembly> assemblies)
    {
        assemblies.Add(typeof(CoreApplicationPart).Assembly);
        return assemblies;
    }
}
