using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Splat;

namespace NnCase.Designer.Commands
{
    public class TargetableCommand : ICommand
    {
        private readonly Command _command;
        private readonly ICommandRouter _commandRouter;
        
        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public TargetableCommand(Command command)
        {
            _command = command;
            _commandRouter = Locator.Current.GetService<ICommandRouter>();
        }

        public bool CanExecute(object parameter)
        {
            var commandHandler = _commandRouter.GetCommandHandlerProxy(_command.CommandDefinition);
            if (commandHandler == null)
                return false;

            commandHandler.Update(_command);
            return true;
        }

        public async void Execute(object parameter)
        {
            try
            {
                var commandHandler = _commandRouter.GetCommandHandlerProxy(_command.CommandDefinition);
                if (commandHandler != null)
                    await commandHandler.Execute(_command);
            }
            catch
            {
            }
        }
    }
}
