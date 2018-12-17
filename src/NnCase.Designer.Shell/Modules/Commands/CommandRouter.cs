using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.Commands
{
    public class CommandRouter : ICommandRouter
    {
        private readonly Dictionary<Type, ICommandHandler> _commandHandlers = new Dictionary<Type, ICommandHandler>();

        public CommandRouter(IEnumerable<ICommandHandler> commandHandlers)
        {
            _commandHandlers = (from handler in commandHandlers
                                from iface in handler.GetType().GetTypeInfo().ImplementedInterfaces
                                where iface.IsConstructedGenericType && iface.GetGenericTypeDefinition() == typeof(ICommandHandler<>)
                                let t = iface.GenericTypeArguments[0]
                                group handler by t).ToDictionary(o => o.Key, o => o.First());
        }

        public ICommandHandler GetCommandHandler(CommandDefinitionBase commandDefinition)
        {
            _commandHandlers.TryGetValue(commandDefinition.GetType(), out var handler);
            return handler;
        }

        public ICommandHandlerProxy GetCommandHandlerProxy(CommandDefinitionBase commandDefinition)
        {
            var handler = GetCommandHandler(commandDefinition);
            if (handler != null)
            {
                var type = typeof(CommandHandlerProxy<>).MakeGenericType(commandDefinition.GetType());
                return (ICommandHandlerProxy)Activator.CreateInstance(type, handler);
            }

            return null;
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
