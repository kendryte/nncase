using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model.Layers;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers
{
    public class OutputLayerViewModel : LayerViewModel
    {
        public InputConnectorViewModel Input { get; }

        public OutputLayerViewModel()
        {
            Input = AddInput("input");
        }

        protected override void BuildModelCore(BuildGraphContext context)
        {
            var model = new OutputLayer(Input.Connection.From.Dimensions) { Name = Name };
            context.InputConnectors[Input] = model.Input;
            context.Layers[this] = new[] { model };
        }
    }
}
