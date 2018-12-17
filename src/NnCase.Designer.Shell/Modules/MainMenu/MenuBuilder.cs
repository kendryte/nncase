using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;
using NnCase.Designer.Modules.MainMenu.Models;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.MainMenu
{
    public class MenuBuilder : IMenuBuilder
    {
        private readonly ICommandService _commandService;
        private readonly MenuBarDefinition[] _menuBars;
        private readonly MenuDefinition[] _menus;
        private readonly MenuItemGroupDefinition[] _menuItemGroups;
        private readonly MenuItemDefinition[] _menuItems;

        public MenuBuilder(ICommandService commandService,
            IEnumerable<MenuBarDefinition> menuBars,
            IEnumerable<MenuDefinition> menus,
            IEnumerable<MenuItemGroupDefinition> menuItemGroups,
            IEnumerable<MenuItemDefinition> menuItems)
        {
            _commandService = commandService;
            _menuBars = menuBars.ToArray();
            _menus = menus.ToArray();
            _menuItemGroups = menuItemGroups.ToArray();
            _menuItems = menuItems.ToArray();
        }

        public void BuildMenuBar(MenuBarDefinition menuBarDefinition, MenuModel result)
        {
            var menus = _menus
                .Where(x => x.MenuBar == menuBarDefinition)
                .OrderBy(x => x.SortOrder);

            foreach (var menu in menus)
            {
                var menuModel = new TextMenuItem(menu);
                AddGroupsRecursive(menu, menuModel);
                if (menuModel.Children.Any())
                    result.Items.Add(menuModel);
            }
        }

        private void AddGroupsRecursive(MenuDefinitionBase menu, StandardMenuItem menuModel)
        {
            var groups = _menuItemGroups
                .Where(x => x.Parent == menu)
                .OrderBy(x => x.SortOrder)
                .ToList();

            for (int i = 0; i < groups.Count; i++)
            {
                var group = groups[i];
                var menuItems = _menuItems
                    .Where(x => x.Group == group)
                    .OrderBy(x => x.SortOrder);

                foreach (var menuItem in menuItems)
                {
                    var menuItemModel = (menuItem.CommandDefinition != null)
                        ? new CommandMenuItem(_commandService.GetCommand(menuItem.CommandDefinition), menuModel)
                        : (StandardMenuItem)new TextMenuItem(menuItem);
                    AddGroupsRecursive(menuItem, menuItemModel);
                    menuModel.Children.Add(menuItemModel);
                }

                if (i < groups.Count - 1 && menuItems.Any())
                    menuModel.Children.Add(MenuItemBase.Separator);
            }
        }
    }
}
