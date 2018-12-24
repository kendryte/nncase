using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.Inspector.Inspectors;

namespace NnCase.Designer.Modules.Inspector.Conventions
{
    public class StandardPropertyEditorBuilder<T, TEditor> : PropertyEditorBuilder
        where TEditor : IEditor, new()
    {
        public override bool IsApplicable(PropertyDescriptor propertyDescriptor)
        {
            return propertyDescriptor.PropertyType == typeof(T);
        }

        public override IEditor BuildEditor(PropertyDescriptor propertyDescriptor)
        {
            return new TEditor();
        }
    }
}
