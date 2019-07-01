using System;
using System.Collections.Generic;

namespace NnCase.IR
{
    public abstract class Connector
    {
        public string Name { get; }

        public Node Owner { get; }

        public DataType Type { get; }

        public Shape Shape { get; }

        public Connector(string name, Node owner, DataType type, Shape shape)
        {
            Name = name;
            Owner = owner;
            Type = type;
            Shape = shape;
        }
    }

    public class InputConnector : Connector
    {
        public OutputConnector Connection { get; private set; }

        public InputConnector(string name, Node owner, DataType type, Shape shape)
            : base(name, owner, type, shape)
        {
        }

        public void Connect(OutputConnector connector)
        {
            if (Type != connector.Type)
                throw new ArgumentException($"Types must be same, while got {Type} and {connector.Type}");

            if (Shape != connector.Shape)
                throw new ArgumentException($"Shapes must be same, while got {Shape} and {connector.Shape}");

            if (Connection != connector)
            {
                ClearConnection();
                Connection = connector;
                connector.Connect(this);
            }
        }

        public void ClearConnection()
        {
            var conn = Connection;
            if (conn != null)
            {
                Connection = null;
                conn.Disconnect(this);
            }
        }
    }

    public class OutputConnector : Connector
    {
        private List<InputConnector> _connections = new List<InputConnector>();

        public IReadOnlyList<InputConnector> Connections => _connections;

        public OutputConnector(string name, Node owner, DataType type, Shape shape)
            : base(name, owner, type, shape)
        {
        }

        public void Connect(InputConnector connector)
        {
            if (!_connections.Contains(connector))
            {
                connector.Connect(this);
                _connections.Add(connector);
            }
        }

        public void Disconnect(InputConnector connector)
        {
            if (_connections.Remove(connector))
                connector.ClearConnection();
        }

        public void ClearConnections()
        {
            var connections = _connections;
            _connections = new List<InputConnector>();

            foreach (var conn in connections)
                conn.ClearConnection();
        }
    }
}
