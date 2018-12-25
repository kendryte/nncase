using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers.K210
{
    public enum K210Stride
    {
        [Display(Name = "1x1")]
        Stride1x1,
        [Display(Name = "2x2")]
        Stride2x2
    }

    public class K210Conv2dViewModel : LayerViewModel<K210Conv2d>
    {
        public InputConnectorViewModel Input { get; }

        public OutputConnectorViewModel Output { get; }

        private ActivationFunctionType _activation;
        public ActivationFunctionType Activation
        {
            get => _activation;
            set
            {
                if (_activation != value)
                {
                    _activation = value;
                    this.RaisePropertyChanged();
                }
            }
        }

        private K210Stride _stride;
        public K210Stride Stride
        {
            get => _stride;
            set
            {
                if (_stride != value)
                {
                    _stride = value;
                    this.RaisePropertyChanged();
                }
            }
        }

        public K210Conv2dViewModel()
        {
            Input = AddInput("input");
            Output = AddOutput("output", new[] { 1, 1, 1, 1 });
        }
    }
}
