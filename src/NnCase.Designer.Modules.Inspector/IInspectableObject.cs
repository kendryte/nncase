using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.Inspector.Inspectors;

namespace NnCase.Designer.Modules.Inspector
{
    public interface IInspectableObject
    {
        IReadOnlyList<IInspector> Inspectors { get; }
    }
}
