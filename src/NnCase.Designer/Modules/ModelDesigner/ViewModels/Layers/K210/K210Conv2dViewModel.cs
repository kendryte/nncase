using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model.Layers;
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

    public enum K210Conv2dKernelSize
    {
        [Display(Name = "1x1")]
        Size1x1,
        [Display(Name = "3x3")]
        Size3x3
    }

    public class K210Conv2dViewModel : LayerViewModel
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

        public K210Conv2dViewModel()
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
            var weights = new DenseTensor<float>(new[] { OutputChannels, Input.Connection.From.Dimensions[1], kernelSize, kernelSize });
            var bias = new DenseTensor<float>(new[] { OutputChannels });
            var model = new K210Conv2d(Input.Connection.From.Dimensions, K210Conv2dType.Conv2d, weights, bias, pool, Activation, null);
            context.InputConnectors[Input] = model.Input;
            context.OutputConnectors[Output] = model.Output;
            context.Layers[this] = new[] { model };
        }
    }
}
