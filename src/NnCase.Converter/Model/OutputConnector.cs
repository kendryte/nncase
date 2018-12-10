using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Model
{
    public sealed class OutputConnector
    {
        private int[] _dimensions;

        public ReadOnlySpan<int> Dimensions => _dimensions;

        public Layer Owner { get; }

        public string Name { get; }

        private readonly List<Connection> _connections = new List<Connection>();

        public IReadOnlyList<Connection> Connections => _connections;

        public OutputConnector(string name, ReadOnlySpan<int> dimensions, Layer owner)
        {
            Name = name;
            _dimensions = dimensions.ToArray();
            Owner = owner;
        }

        public Connection AddConnection(InputConnector to)
        {
            var conn = _connections.FirstOrDefault(o => o.To == to);
            if (conn != null) return conn;
            conn = new Connection(this, to);
            _connections.Add(conn);
            to.SetConnection(this);

            //if (_connections.Count > 1)
            //    throw new InvalidOperationException();
            return conn;
        }

        public void RemoveConnection(InputConnector to)
        {
            _connections.RemoveAll(o => o.To == to);
            to.ClearConnection();
        }
    }
}
