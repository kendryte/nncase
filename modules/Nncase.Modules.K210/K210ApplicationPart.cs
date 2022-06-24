using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// K210 application part extensions.
/// </summary>
public static class K210ApplicationPart
{
    /// <summary>
    /// Add k210 assembly.
    /// </summary>
    /// <param name="assemblies">Assembly collection.</param>
    /// <returns>Updated assembly collection.</returns>
    public static IList<Assembly> AddK210(this IList<Assembly> assemblies)
    {
        assemblies.Add(typeof(K210ApplicationPart).Assembly);
        return assemblies;
    }
}
