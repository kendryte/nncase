using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using NnCase.Designer.Commands;
using NnCase.Designer.Modules.ModelDesigner.ViewModels;
using NnCase.Designer.Modules.ModelDesigner.ViewModels.Layers;
using NnCase.Designer.Services;
using ReactiveUI;
using Splat;

namespace NnCase.Designer.Modules.ModelDesigner.Commands
{
    public class ExportScriptCommandHandler : ICommandHandler<ExportScriptCommandDefinition>
    {
        private readonly IShell _shell;

        public ExportScriptCommandHandler(IShell shell)
        {
            _shell = shell;
        }

        public Task Execute(Command command)
        {
            if (_shell.ActiveDocument is GraphViewModel graph)
            {
                var context = new BuildGraphContext();
                graph.Build(context);
            }

            return Task.CompletedTask;
        }

        public void Update(Command command)
        {
        }
    }
}
