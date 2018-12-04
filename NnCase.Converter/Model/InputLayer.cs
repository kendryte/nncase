using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model
{
    public class InputLayer : Layer
    {
        public OutputConnector Output { get; }

        public InputLayer(ReadOnlySpan<int> dimensions)
        {
            Output = AddOutput("output", dimensions);
        }
    }
}
