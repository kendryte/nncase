using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Services
{
    public interface ICommandService
    {
        Command GetCommand(CommandDefinitionBase commandDefinition);

        TargetableCommand GetTargetableCommand(Command command);
    }
}
