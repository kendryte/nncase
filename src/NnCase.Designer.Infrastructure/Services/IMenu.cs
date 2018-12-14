using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;

namespace NnCase.Designer.Services
{
    public interface IMenu
    {
        ObservableCollection<MenuItemBase> Items { get; }
    }
}
