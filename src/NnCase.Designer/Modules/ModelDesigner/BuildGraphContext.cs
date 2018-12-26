using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using NnCase.Designer.Modules.ModelDesigner.ViewModels;

namespace NnCase.Designer.Modules.ModelDesigner
{
    public class BuildGraphContext
    {
        public Graph Graph { get; set; }

        public Dictionary<LayerViewModel, IList<Layer>> Layers { get; } = new Dictionary<LayerViewModel, IList<Layer>>();

        public Dictionary<InputConnectorViewModel, InputConnector> InputConnectors { get; } = new Dictionary<InputConnectorViewModel, InputConnector>();

        public Dictionary<OutputConnectorViewModel, OutputConnector> OutputConnectors { get; } = new Dictionary<OutputConnectorViewModel, OutputConnector>();
    }
}
