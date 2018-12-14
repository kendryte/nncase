using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ReactiveUI;

namespace NnCase.Designer.Menus
{
    public abstract class MenuItemBase : ReactiveObject
    {
        public static MenuItemBase Separator { get; } = new MenuItemSeparator();

        public ObservableCollection<MenuItemBase> Children { get; } = new ObservableCollection<MenuItemBase>();
    }

    public class MenuItemSeparator : MenuItemBase
    {
    }
}
