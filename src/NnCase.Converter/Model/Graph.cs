using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model
{
    public class Graph
    {
        public IReadOnlyList<InputConnector> Inputs { get; }

        public IReadOnlyList<OutputConnector> Outputs { get; }

        public Graph(IReadOnlyList<InputConnector> inputs, IReadOnlyList<OutputConnector> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}
