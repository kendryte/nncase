using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace NnCase.Designer.Menus
{
    public class TextMenuItem : StandardMenuItem
    {
        private readonly MenuDefinitionBase _menuDefinition;

        public override string Text => _menuDefinition.Text;

        public override ICommand Command => null;

        public TextMenuItem(MenuDefinitionBase menuDefinition)
        {
            _menuDefinition = menuDefinition;
        }
    }
}
