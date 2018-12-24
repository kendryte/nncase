using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Modules.Inspector.Inspectors
{
    public class CollapsibleGroupBuilder : InspectorBuilder<CollapsibleGroupBuilder>
    {
        internal CollapsibleGroupViewModel ToCollapsibleGroup(string name)
        {
            return new CollapsibleGroupViewModel(name, Inspectors);
        }
    }
}
