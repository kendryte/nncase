using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Model
{
    public sealed class InputConnector
    {
        private int[] _dimensions;

        public ReadOnlySpan<int> Dimensions => _dimensions;

        public Layer Owner { get; }

        public string Name { get; }

        public Connection Connection { get; private set; }

        public InputConnector(string name, ReadOnlySpan<int> dimensions, Layer owner)
        {
            Name = name;
            _dimensions = dimensions.ToArray();
            Owner = owner;
        }

        public Connection SetConnection(OutputConnector from)
        {
            if (!from.Dimensions.SequenceEqual(Dimensions))
                throw new InvalidOperationException("Dimensions must be equal.");

            if (Connection != null)
            {
                if (Connection.From == from)
                    return Connection;
                else
                    ClearConnection();
            }

            Connection = new Connection(from, this);
            from.AddConnection(this);
            return Connection;
        }

        public void ClearConnection()
        {
            if (Connection != null)
            {
                var from = Connection.From;
                Connection = null;
                from.RemoveConnection(this);
            }
        }
    }
}
