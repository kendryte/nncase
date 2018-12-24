using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Modules.Inspector
{
    public class InspectableObjectBuilder : InspectorBuilder<InspectableObjectBuilder>
    {
        public InspectableObject ToInspectableObject()
        {
            return new InspectableObject(Inspectors);
        }
    }
}
