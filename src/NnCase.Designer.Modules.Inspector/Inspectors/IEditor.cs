using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Modules.Inspector.Inspectors
{
    public interface IEditor : IInspector
    {
        BoundPropertyDescriptor BoundPropertyDescriptor { get; set; }
        bool CanReset { get; }
        void Reset();
    }
}
