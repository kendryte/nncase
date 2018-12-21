using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace NnCase.Designer.Modules.Shell.Controls
{
    public class PanesTemplateSelector : DataTemplateSelector
    {
        public DataTemplate DocumentTemplate { get; set; }

        public DataTemplate ToolTemplate { get; set; }

        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            switch (item)
            {
                case IDocument _:
                    return DocumentTemplate;
                case ITool _:
                    return ToolTemplate;
                default:
                    return base.SelectTemplate(item, container);
            }
        }
    }
}
