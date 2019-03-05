using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model.Layers;
using ReactiveUI;

namespace NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers.K210
{
    public class K210SeparableConv2dViewModel : LayerViewModel
    {
        public InputConnectorViewModel Input { get; }

        public OutputConnectorViewModel Output { get; }

        private ActivationFunctionType _activation = ActivationFunctionType.Relu;
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
                    UpdateOutput();
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
                    UpdateOutput();
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
                    UpdateOutput();
                }
            }
        }

        public K210SeparableConv2dViewModel()
        {
            Input = AddInput("input");
            Input.Updated += Input_Updated;
            Output = AddOutput("output", new[] { 1, 1, 1, 1 });
        }

        private void Input_Updated(object sender, EventArgs e)
        {
            UpdateOutput();
        }

        private void UpdateOutput()
        {
            var input = Input.Connection?.From;
            if (input != null)
            {
                var stride = Stride == K210Stride.Stride1x1 ? 1 : 2;
                Output.SetDimension(d =>
                {
                    d[1] = OutputChannels;
                    d[2] = input.Dimensions[2] / stride;
                    d[3] = input.Dimensions[3] / stride;
                });
            }
        }

        protected override void BuildModelCore(BuildGraphContext context)
        {
            var kernelSize = KernelSize == K210Conv2dKernelSize.Size1x1 ? 1 : 3;
            var pool = Stride == K210Stride.Stride1x1 ? K210PoolType.None : K210PoolType.LeftTop;
            var dwWeights = new DenseTensor<float>(new[] { 1, Input.Connection.From.Dimensions[1], kernelSize, kernelSize });
            var pwWeights = new DenseTensor<float>(new[] { OutputChannels, Input.Connection.From.Dimensions[1], 1, 1 });
            var bias = new DenseTensor<float>(new[] { OutputChannels });

            var model = new K210SeparableConv2d(Input.Connection.From.Dimensions, dwWeights, pwWeights, bias, pool, Activation);
            context.InputConnectors[Input] = model.Input;
            context.OutputConnectors[Output] = model.Output;
            context.Layers[this] = new[] { model };
        }
    }
}
