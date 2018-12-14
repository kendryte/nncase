using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;
using NnCase.Designer.Services;
using ReactiveUI;

namespace NnCase.Designer.Modules.MainMenu.Models
{
    public class MenuModel : ReactiveObject, IMenu
    {
        public ObservableCollection<MenuItemBase> Items { get; } = new ObservableCollection<MenuItemBase>();
    }
}
