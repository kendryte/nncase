using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer
{
    public static class InspectorModuleExtensions
    {
        public static ICollection<Assembly> AddInspector(this ICollection<Assembly> assemblies)
        {
            assemblies.Add(typeof(InspectorModuleExtensions).Assembly);
            return assemblies;
        }
    }
}
