using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.Inspector.Inspectors;

namespace NnCase.Designer.Modules.Inspector
{
    public class InspectableObject : IInspectableObject
    {
        public IReadOnlyList<IInspector> Inspectors { get; }

        public InspectableObject(IReadOnlyList<IInspector> inspectors)
        {
            Inspectors = inspectors;
        }
    }
}
