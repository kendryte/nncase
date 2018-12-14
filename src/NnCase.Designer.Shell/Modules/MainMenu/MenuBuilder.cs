using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Menus;
using NnCase.Designer.Modules.MainMenu.Models;

namespace NnCase.Designer.Modules.MainMenu
{
    public class MenuBuilder : IMenuBuilder
    {
        private readonly MenuBarDefinition[] _menuBars;
        private readonly MenuDefinition[] _menus;
        private readonly MenuItemGroupDefinition[] _menuItemGroups;
        private readonly MenuItemDefinition[] _menuItems;

        public MenuBuilder(IEnumerable<MenuBarDefinition> menuBars,
            IEnumerable<MenuDefinition> menus,
            IEnumerable<MenuItemGroupDefinition> menuItemGroups,
            IEnumerable<MenuItemDefinition> menuItems)
        {
            _menuBars = menuBars.ToArray();
            _menus = menus.ToArray();
            _menuItemGroups = menuItemGroups.ToArray();
            _menuItems = menuItems.ToArray();
        }

        public void BuildMenuBar(MenuBarDefinition menuBarDefinition, MenuModel menuModel)
        {
            var menus = _menus
                .Where(x => x.MenuBar == menuBarDefinition)
                .OrderBy(x => x.SortOrder);

            foreach (var menu in menus)
            {
                var model = new TextMenuItem(menu);
                AddGroupsRecursive(menu, model);
                if (model.Children.Any())
                    menuModel.Items.Add(model);
            }
        }

        private void AddGroupsRecursive(MenuDefinition menu, TextMenuItem model)
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
                    //var menuItemModel = (menuItem.CommandDefinition != null)
                    //    ? new CommandMenuItem(_commandService.GetCommand(menu.CommandDefinition), model)
                    //    : (StandardMenuItem)new TextMenuItem(menu);
                    //AddGroupsRecursive(menuItem, menuItemModel);
                    //model.Children.Add(menuItemModel);
                }

                if (i < groups.Count - 1 && menuItems.Any())
                    model.Children.Add(MenuItemBase.Separator);
            }
        }
    }
}
