using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using NnCase.Designer.Menus;

namespace NnCase.Designer.Modules.MainMenu.Controls
{
    public class MenuEx : Menu
    {
        private object _currentItem;

        protected override bool IsItemItsOwnContainerOverride(object item)
        {
            _currentItem = item;
            return base.IsItemItsOwnContainerOverride(item);
        }

        protected override DependencyObject GetContainerForItemOverride()
        {
            switch (_currentItem)
            {
                case MenuItemSeparator _:
                    return new Separator();
                default:
                    return base.GetContainerForItemOverride();
            }
        }
    }
}
