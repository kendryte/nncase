using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;
using NnCase.Designer.Modules.ModelDesigner.Commands;

namespace NnCase.Designer.Modules.ModelDesigner
{
    public static class MenuDefinitions
    {
        public static MenuItemDefinition OpenGraphMenuItem = new CommandMenuItemDefinition<OpenGraphCommandDefinition>(
            MainMenu.MenuDefinictions.FileNewOpenMenuGroup, 2);
        public static MenuItemDefinition ExportScriptMenuItem = new CommandMenuItemDefinition<ExportScriptCommandDefinition>(
            MainMenu.MenuDefinictions.FileNewOpenMenuGroup, 3);
        public static MenuItemDefinition ExportK210CodeMenuItem = new CommandMenuItemDefinition<ExportK210CodeCommandDefinition>(
            MainMenu.MenuDefinictions.FileNewOpenMenuGroup, 3);
    }
}
