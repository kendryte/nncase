using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.ModelDesigner
{
    public class ModelDesignerModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.Register(c => MenuDefinitions.OpenGraphMenuItem)
                .PreserveExistingDefaults();

            builder.RegisterType<Commands.OpenGraphCommandDefinition>()
                .As<CommandDefinitionBase>();
            builder.RegisterType<Commands.OpenGraphCommandHandler>()
                .As<ICommandHandler>()
                .PreserveExistingDefaults();

            builder.RegisterInstance(Toolbox.ToolboxItems.InputLayer)
                .PreserveExistingDefaults();
            builder.RegisterInstance(Toolbox.ToolboxItems.K210Conv2d)
                .PreserveExistingDefaults();
            builder.RegisterInstance(Toolbox.ToolboxItems.K210SeparableConv2d)
                .PreserveExistingDefaults();

            builder.RegisterType<ViewModels.GraphViewModel>();
            builder.RegisterType<Views.GraphView>();
        }
    }
}
