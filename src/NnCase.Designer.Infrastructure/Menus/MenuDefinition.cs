using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Menus
{
    public class MenuDefinition : MenuDefinitionBase
    {
        public MenuBarDefinition MenuBar { get; }

        public override int SortOrder { get; }

        public override string Text { get; }

        public override Uri IconSource => null;

        public override KeyGesture KeyGesture => null;

        public override CommandDefinitionBase CommandDefinition => null;

        public MenuDefinition(MenuBarDefinition menuBar, int sortOrder, string text)
        {
            MenuBar = menuBar;
            SortOrder = sortOrder;
            Text = text;
        }
    }
}
