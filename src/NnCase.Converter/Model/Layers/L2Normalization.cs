using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class L2Normalization : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public L2Normalization(ReadOnlySpan<int> dimensions)
        {
            Input = AddInput("input", dimensions);
            Output = AddOutput("output", dimensions);
        }
    }
}
