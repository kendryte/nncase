using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ReactiveUI;

namespace NnCase.Designer.Modules.Inspector.Inspectors
{
    public abstract class InspectorBase : ReactiveObject, IInspector
    {
        public abstract string Name { get; }

        public abstract bool IsReadOnly { get; }
    }
}
