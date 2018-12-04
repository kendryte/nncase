using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model
{
    public abstract class Layer
    {
        private readonly List<InputConnector> _inputConnectors = new List<InputConnector>();
        private readonly List<OutputConnector> _outputConnectors = new List<OutputConnector>();

        public InputConnector AddInput(string name, ReadOnlySpan<int> dimensions)
        {
            var conn = new InputConnector(name, dimensions, this);
            _inputConnectors.Add(conn);
            return conn;
        }

        public OutputConnector AddOutput(string name, ReadOnlySpan<int> dimensions)
        {
            var conn = new OutputConnector(name, dimensions, this);
            _outputConnectors.Add(conn);
            return conn;
        }
    }
}
