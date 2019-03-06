using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Converters
{
    public class LayerNotSupportedException : Exception
    {
        public LayerNotSupportedException(string layer)
            :base($"Layer {layer} is not supported")
        {

        }

        public LayerNotSupportedException(string layer, string reason)
            : base($"Layer {layer} is not supported: {reason}")
        {

        }
    }
}
