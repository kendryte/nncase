using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;

namespace NnCase.Designer.Modules.MainMenu
{
    public static class MenuDefinictions
    {
        public static MenuBarDefinition MainMenuBar = new MenuBarDefinition();

        public static MenuDefinition FileMenu = new MenuDefinition(MainMenuBar, 0, "File");

        public static MenuItemGroupDefinition FileNewOpenMenuGroup = new MenuItemGroupDefinition(FileMenu, 0);
    }
}
