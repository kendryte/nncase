using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Menus
{
    public abstract class MenuItemDefinition : MenuDefinitionBase
    {
        public MenuItemGroupDefinition Group { get; }

        public override int SortOrder { get; }

        public MenuItemDefinition(MenuItemGroupDefinition group, int sortOrder)
        {
            Group = group;
            SortOrder = sortOrder;
        }
    }
}
