using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.Inspector.Inspectors;

namespace NnCase.Designer.Modules.Inspector.Conventions
{
    public abstract class PropertyEditorBuilder
    {
        public abstract bool IsApplicable(PropertyDescriptor propertyDescriptor);
        public abstract IEditor BuildEditor(PropertyDescriptor propertyDescriptor);
    }
}
