using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Commands
{
    public interface ICommandRouter
    {
        ICommandHandler GetCommandHandler(CommandDefinitionBase commandDefinition);

        ICommandHandlerProxy GetCommandHandlerProxy(CommandDefinitionBase commandDefinition);
    }
}
