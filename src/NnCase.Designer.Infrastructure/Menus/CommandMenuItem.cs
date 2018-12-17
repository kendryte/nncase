using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using NnCase.Designer.Commands;
using NnCase.Designer.Services;
using Splat;

namespace NnCase.Designer.Menus
{
    public class CommandMenuItem : StandardMenuItem
    {
        private readonly Command _command;
        private readonly StandardMenuItem _parent;

        public override string Text => _command.Text;

        public override ICommand Command => Locator.Current.GetService<ICommandService>().GetTargetableCommand(_command);

        public CommandMenuItem(Command command, StandardMenuItem parent)
        {
            _command = command;
            _parent = parent;
        }
    }
}
