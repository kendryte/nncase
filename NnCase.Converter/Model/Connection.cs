using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model
{
    public sealed class Connection
    {
        public OutputConnector From { get; }

        public InputConnector To { get; }

        public Connection(OutputConnector from, InputConnector to)
        {
            From = from ?? throw new ArgumentNullException(nameof(from));
            To = to ?? throw new ArgumentNullException(nameof(to));
        }
    }
}
