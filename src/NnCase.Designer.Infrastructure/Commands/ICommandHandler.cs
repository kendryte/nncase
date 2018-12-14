using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Commands
{
    public interface ICommandHandler
    {
    }

    public interface ICommandHandler<TCommandDefinition> : ICommandHandler
        where TCommandDefinition : CommandDefinition
    {
        void Update(Command command);

        Task Execute(Command command);
    }

    public interface ICommandHandlerProxy
    {
        void Update(Command command);

        Task Execute(Command command);
    }
}
