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
    public class CommandMenuItemDefinition<TCommandDefinition> : MenuItemDefinition
        where TCommandDefinition : CommandDefinitionBase
    {
        public override string Text => CommandDefinition.Text;

        public override Uri IconSource => CommandDefinition.IconSource;

        public override KeyGesture KeyGesture => null;

        public override CommandDefinitionBase CommandDefinition { get; }

        public CommandMenuItemDefinition(MenuItemGroupDefinition group, int sortOrder)
            :base(group, sortOrder)
        {
            CommandDefinition = Locator.Current.GetService<ICommandService>().GetCommandDefinition(typeof(TCommandDefinition));
        }
    }
}
