using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using NnCase.Designer.Modules.Inspector.Inspectors;

namespace NnCase.Designer.Modules.Inspector.Controls
{
    public class InspectorItemTemplateSelector : DataTemplateSelector
    {
        public DataTemplate LabelledTemplate { get; set; }

        public DataTemplate DefaultTemplate { get; set; }

        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            if (item is ILabelledInspector)
                return LabelledTemplate;
            return DefaultTemplate;
        }
    }
}
