using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model.Layers;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers
{
    public class InputLayerViewModel : LayerViewModel<InputLayer>
    {
        public OutputConnectorViewModel Output { get; }

        public int Width
        {
            get => Output.Dimensions[3];
            set => Output.SetDimension(3, value);
        }

        public int Height
        {
            get => Output.Dimensions[2];
            set => Output.SetDimension(2, value);
        }

        public int Channels => Output.Dimensions[1];

        public InputLayerViewModel()
        {
            Output = AddOutput("output", new[] { 1, 3, 128, 128 });
        }
    }
}
