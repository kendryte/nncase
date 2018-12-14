using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.Commands
{
    public class CommandRouter : ICommandRouter
    {
        private readonly Dictionary<Type, HashSet<ICommandHandler>> _commandHandlers = new Dictionary<Type, HashSet<ICommandHandler>>();

        public CommandRouter(IEnumerable<ICommandHandler> commandHandlers)
        {

        }

        public ICommandHandler GetCommandHandler(CommandDefinitionBase commandDefinition)
        {
            throw new NotImplementedException();
        }

        public ICommandHandlerProxy GetCommandHandlerProxy(CommandDefinitionBase commandDefinition)
        {
            var handler = GetCommandHandler(commandDefinition);
            var type = typeof(CommandHandlerProxy<>).MakeGenericType(commandDefinition.GetType());
            return (ICommandHandlerProxy)Activator.CreateInstance(type, handler);
        }

        private class CommandHandlerProxy<TCommandDefinition> : ICommandHandlerProxy
            where TCommandDefinition : CommandDefinition
        {
            private readonly ICommandHandler<TCommandDefinition> _handler;

            public CommandHandlerProxy(ICommandHandler handler)
            {
                _handler = (ICommandHandler<TCommandDefinition>)handler;
            }

            public void Update(Command command)
                => _handler.Update(command);

            public Task Execute(Command command)
                => _handler.Execute(command);
        }
    }
}
