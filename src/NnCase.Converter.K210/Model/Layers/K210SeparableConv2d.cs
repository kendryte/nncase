using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Model.Layers
{
    public class K210SeparableConv2d : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Tensor<float> DwWeights { get; }

        public Tensor<float> PwWeights { get; }

        public Tensor<float> Bias { get; }

        public K210PoolType PoolType { get; }

        public ActivationFunctionType FusedActivationFunction { get; }

        public int KernelWidth => DwWeights.Dimensions[3];

        public int KernelHeight => DwWeights.Dimensions[2];

        public int InputChannels => DwWeights.Dimensions[1];

        public int OutputChannels => PwWeights.Dimensions[0];

        public K210SeparableConv2d(ReadOnlySpan<int> dimensions, Tensor<float> dwWeights, Tensor<float> pwWeigths, Tensor<float> bias, K210PoolType poolType, ActivationFunctionType fusedActivationFunction)
        {
            if (dimensions[2] < 4 || dimensions[3] < 4)
                throw new ArgumentOutOfRangeException("Lower than 4x4 input is not supported in dwConv2d.");
            
            PoolType = poolType;
            FusedActivationFunction = fusedActivationFunction;
            DwWeights = dwWeights;
            PwWeights = pwWeigths;
            Bias = bias;

            var stride = GetStride();

            if (dimensions[2] / stride < 4 || dimensions[3] / stride < 4)
                throw new ArgumentOutOfRangeException("Lower than 4x4 output is not supported in dwConv2d.");

            Input = AddInput("input", dimensions);
            Output = AddOutput("output", new[] {
                dimensions[0],
                OutputChannels,
                dimensions[2] / stride,
                dimensions[3] / stride
            });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            throw new NotSupportedException();
        }

        private int GetStride()
        {
            int stride;
            switch (PoolType)
            {
                case K210PoolType.None:
                    stride = 1;
                    break;
                case K210PoolType.LeftTop:
                    stride = 2;
                    break;
                case K210PoolType.MaxPool2x2:
                    stride = 2;
                    break;
                case K210PoolType.MaxPool4x4:
                    stride = 4;
                    break;
                default:
                    throw new NotSupportedException();
            }

            return stride;
        }
    }
}
