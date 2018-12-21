using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace NnCase.Designer.Modules.Shell.Controls
{
    public class PanesStyleSelector : StyleSelector
    {
        public Style DocumentStyle { get; set; }

        public Style ToolStyle { get; set; }

        public override Style SelectStyle(object item, DependencyObject container)
        {
            switch (item)
            {
                case IDocument _:
                    return DocumentStyle;
                case ITool _:
                    return ToolStyle;
                default:
                    return base.SelectStyle(item, container);
            }
        }
    }
}
