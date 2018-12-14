using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ReactiveUI;

namespace NnCase.Designer.Commands
{
    public class Command : ReactiveObject
    {
        public CommandDefinitionBase CommandDefinition { get; }

        private string _text;
        public string Text
        {
            get => _text;
            set => this.RaiseAndSetIfChanged(ref _text, value);
        }

        public Command(CommandDefinitionBase commandDefinition)
        {
            CommandDefinition = commandDefinition;
            Text = CommandDefinition.Text;
        }
    }
}
