using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels
{
    public class InputConnectorViewModel : ReactiveObject
    {
        public InputConnector Model { get; }

        public string Name => Model.Name;

        public ReadOnlySpan<int> Dimensions => Model.Dimensions;

        public ILayerViewModel Owner { get; }

        public InputConnectorViewModel(string name, ReadOnlySpan<int> dimensions, ILayerViewModel owner)
        {
            Model = new InputConnector(name, dimensions, owner.Model);
            Owner = owner;
        }
    }
}
