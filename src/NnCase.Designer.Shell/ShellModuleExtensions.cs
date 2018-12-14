using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer
{
    public static class ShellModuleExtensions
    {
        public static ICollection<Assembly> AddShell(this ICollection<Assembly> assemblies)
        {
            assemblies.Add(typeof(ShellModuleExtensions).Assembly);
            return assemblies;
        }
    }
}
