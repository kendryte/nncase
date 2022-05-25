using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// StackVM application part extensions.
/// </summary>
public static class StackVMApplicationPart
{
    /// <summary>
    /// Add stackVM assembly.
    /// </summary>
    /// <param name="assemblies">Assembly collection.</param>
    /// <returns>Updated assembly collection.</returns>
    public static IList<Assembly> AddStackVM(this IList<Assembly> assemblies)
    {
        assemblies.Add(typeof(StackVMApplicationPart).Assembly);
        return assemblies;
    }
}
