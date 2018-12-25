using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers.K210
{
    public class K210SeparableConv2dViewModel : LayerViewModel<K210SeparableConv2d>
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

        private K210Conv2dKernelSize _kernelSize;
        public K210Conv2dKernelSize KernelSize
        {
            get => _kernelSize;
            set
            {
                if (_kernelSize != value)
                {
                    _kernelSize = value;
                    this.RaisePropertyChanged();
                }
            }
        }

        private int _outputChannels = 1;
        public int OutputChannels
        {
            get => _outputChannels;
            set
            {
                if (_outputChannels != value)
                {
                    _outputChannels = value;
                    this.RaisePropertyChanged();
                }
            }
        }

        public K210SeparableConv2dViewModel()
        {
            Input = AddInput("input");
            Output = AddOutput("output", new[] { 1, 1, 1, 1 });
        }
    }
}
