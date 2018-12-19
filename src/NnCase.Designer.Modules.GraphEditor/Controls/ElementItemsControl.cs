using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace NnCase.Designer.Modules.GraphEditor.Controls
{
    public class ElementItemsControl : ListBox
    {
        protected override DependencyObject GetContainerForItemOverride()
        {
            return new ElementItem();
        }

        protected override bool IsItemItsOwnContainerOverride(object item)
        {
            return item is ElementItem;
        }

        public ElementItemsControl()
        {
            SelectionMode = SelectionMode.Extended;
        }
    }
}
