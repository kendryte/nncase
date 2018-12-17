using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.Commands
{
    public class CommandService : ICommandService
    {
        private readonly Dictionary<CommandDefinitionBase, Command> _commands = new Dictionary<CommandDefinitionBase, Command>();
        private readonly Dictionary<Command, TargetableCommand> _targetableCommands = new Dictionary<Command, TargetableCommand>();
        private readonly Dictionary<Type, CommandDefinitionBase> _commandDefinitions = new Dictionary<Type, CommandDefinitionBase>();

        public CommandService(IEnumerable<CommandDefinitionBase> commandDefinitions)
        {
            _commandDefinitions = commandDefinitions.ToDictionary(o => o.GetType(), o => o);
        }

        public CommandDefinitionBase GetCommandDefinition(Type commandDefinitionType)
        {
            _commandDefinitions.TryGetValue(commandDefinitionType, out var commandDefinition);
            return commandDefinition;
        }

        public Command GetCommand(CommandDefinitionBase commandDefinition)
        {
            if (!_commands.TryGetValue(commandDefinition, out var command))
                command = _commands[commandDefinition] = new Command(commandDefinition);
            return command;
        }

        public TargetableCommand GetTargetableCommand(Command command)
        {
            if (!_targetableCommands.TryGetValue(command, out var targetableCommand))
                targetableCommand = _targetableCommands[command] = new TargetableCommand(command);
            return targetableCommand;
        }
    }
}
