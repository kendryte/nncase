using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.ModelDesigner.Commands
{
    public class ExportK210CodeCommandHandler : ICommandHandler<ExportK210CodeCommandDefinition>
    {
        public Task Execute(Command command)
        {
            var dlg = new Views.ExportK210CodeView();
            dlg.ShowDialog();
            return Task.CompletedTask;
        }

        public void Update(Command command)
        {
        }
    }
}
