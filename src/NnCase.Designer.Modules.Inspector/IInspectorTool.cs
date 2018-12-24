using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Modules.Inspector
{
    public interface IInspectorTool : ITool
    {
        IInspectableObject SelectedObject { get; set; }
    }
}
