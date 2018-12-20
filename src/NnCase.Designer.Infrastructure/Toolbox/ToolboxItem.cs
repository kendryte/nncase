using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Toolbox
{
    public class ToolboxItem
    {
        public Type DocumentType { get; set; }

        public string Category { get; set; }

        public string Text { get; set; }

        public Type ItemType { get; set; }
    }
}
