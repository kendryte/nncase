using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.ModelDesigner.ViewModels;
using NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer.Modules.ModelDesigner.Toolbox
{
    public static class ToolboxItems
    {
        public static readonly ToolboxItem InputLayer = new ToolboxItem { Text = "Input Layer", DocumentType = typeof(GraphViewModel), Category = "Layers", ItemType = typeof(InputLayerViewModel) };
    }
}
