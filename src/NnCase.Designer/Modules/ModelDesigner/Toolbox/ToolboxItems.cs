using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.ModelDesigner.ViewModels;
using NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers;
using NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers.K210;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer.Modules.ModelDesigner.Toolbox
{
    public static class ToolboxItems
    {
        public static readonly ToolboxItem InputLayer = new ToolboxItem { Text = "Input Layer", DocumentType = typeof(GraphViewModel), Category = "General", ItemType = typeof(InputLayerViewModel) };
        public static readonly ToolboxItem OutputLayer = new ToolboxItem { Text = "Output Layer", DocumentType = typeof(GraphViewModel), Category = "General", ItemType = typeof(OutputLayerViewModel) };
        public static readonly ToolboxItem K210Conv2d = new ToolboxItem { Text = "Conv2d", DocumentType = typeof(GraphViewModel), Category = "K210", ItemType = typeof(K210Conv2dViewModel) };
        public static readonly ToolboxItem K210SeparableConv2d = new ToolboxItem { Text = "Separable Conv2d", DocumentType = typeof(GraphViewModel), Category = "K210", ItemType = typeof(K210SeparableConv2dViewModel) };
    }
}
