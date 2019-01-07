using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model.Layers;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers
{
    public class InputLayerViewModel : LayerViewModel
    {
        public OutputConnectorViewModel Output { get; }

        private int _width;
        public int Width
        {
            get => _width;
            set
            {
                if (_width != value)
                {
                    _width = value;
                    this.RaisePropertyChanged();
                    UpdateOutput();
                }
            }
        }

        private int _height;
        public int Height
        {
            get => _height;
            set
            {
                if (_height != value)
                {
                    _height = value;
                    this.RaisePropertyChanged();
                    UpdateOutput();
                }
            }
        }

        public int Channels => 3;

        public InputLayerViewModel()
        {
            _width = 128;
            _height = 128;
            Output = AddOutput("output", new[] { 1, Channels, Height, Width });
        }

        private void UpdateOutput()
        {
            Output.SetDimension(d =>
            {
                d[2] = Height;
                d[3] = Width;
            });
        }

        protected override void BuildModelCore(BuildGraphContext context)
        {
            var model = new InputLayer(Output.Dimensions) { Name = Name };
            context.OutputConnectors[Output] = model.Output;
            context.Layers[this] = new[] { model };
        }
    }
}
