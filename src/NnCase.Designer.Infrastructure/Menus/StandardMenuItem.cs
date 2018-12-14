using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace NnCase.Designer.Menus
{
    public abstract class StandardMenuItem : MenuItemBase
    {
        public abstract string Text { get; }

        public abstract ICommand Command { get; }
    }
}
