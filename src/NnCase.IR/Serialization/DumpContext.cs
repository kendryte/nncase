using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Serialization
{
    public class DumpContext
    {
        public string Title { get; set; }

        public Dictionary<string, string> Attributes { get; } = new Dictionary<string, string>();
    }
}
