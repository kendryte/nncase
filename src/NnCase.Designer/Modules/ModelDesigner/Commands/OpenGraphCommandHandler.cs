using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;
using NnCase.Designer.Modules.ModelDesigner.ViewModels;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.ModelDesigner.Commands
{
    public class OpenGraphCommandHandler : ICommandHandler<OpenGraphCommandDefinition>
    {
        private readonly IShell _shell;

        public OpenGraphCommandHandler(IShell shell)
        {
            _shell = shell;
        }

        public Task Execute(Command command)
        {
            _shell.OpenDocument(new GraphViewModel("Graph 1"));
            return Task.CompletedTask;
        }

        public void Update(Command command)
        {

        }
    }
}
