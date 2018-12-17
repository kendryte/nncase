using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;
using NnCase.Designer.Modules.MainMenu.Models;

namespace NnCase.Designer.Modules.MainMenu
{
    public interface IMenuBuilder
    {
        void BuildMenuBar(MenuBarDefinition menuBarDefinition, MenuModel result);
    }
}
