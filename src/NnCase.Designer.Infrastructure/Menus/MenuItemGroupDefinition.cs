using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Menus
{
    public class MenuItemGroupDefinition
    {
        public MenuDefinitionBase Parent { get; }

        public int SortOrder { get; }

        public MenuItemGroupDefinition(MenuDefinitionBase parent, int sortOrder)
        {
            Parent = parent;
            SortOrder = sortOrder;
        }
    }
}
