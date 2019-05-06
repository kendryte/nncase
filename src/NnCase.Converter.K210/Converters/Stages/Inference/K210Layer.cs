using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Inference
{
    public enum K210LayerType
    {
        Invalid = 0,
        Add,
        QuantizedAdd,
        GlobalMaxPool2d,
        QuantizedGlobalMaxPool2d,
        GlobalAveragePool2d,
        QuantizedGlobalAveragePool2d,
        MaxPool2d,
        QuantizedMaxPool2d,
        AveragePool2d,
        QuantizedAveragePool2d,
        Quantize,
        Dequantize,
        Requantize,
        L2Normalization,
        Softmax,
        Concatenation,
        QuantizedConcatenation,
        FullyConnected,
        QuantizedFullyConnected,
        TensorflowFlatten,
        QuantizedTensorflowFlatten,
        ResizeNearestNeighbor,
        QuantizedResizeNearestNeighbor,
        ChannelwiseDequantize,
        Logistic,
        K210Conv = 10240,
        K210AddPadding,
        K210RemovePadding,
        K210Upload
    }

    public class K210LayerHeader
    {
        public K210LayerType Type { get; set; }

        public uint BodySize { get; set; }
    }

    public class K210Layer
    {
        public K210LayerHeader Header { get; set; }

        public object Body { get; set; }
    }
}
