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

        private readonly List<Connection> _connections = new List<Connection>();

        public IReadOnlyList<Connection> Connections => _connections;

        public InputConnector(string name, ReadOnlySpan<int> dimensions, Layer owner)
        {
            Name = name;
            _dimensions = dimensions.ToArray();
            Owner = owner;
        }

        public Connection AddConnection(OutputConnector from)
        {
            var conn = _connections.FirstOrDefault(o => o.From == from);
            if (conn != null) return conn;
            conn = new Connection(from, this);
            _connections.Add(conn);
            from.AddConnection(this);
            return conn;
        }

        public void RemoveConnection(OutputConnector from)
        {
            _connections.RemoveAll(o => o.From == from);
            from.RemoveConnection(this);
        }
    }
}
