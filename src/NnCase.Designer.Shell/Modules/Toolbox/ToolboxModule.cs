using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;
using NnCase.Designer.Commands;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.Toolbox
{
    public class ToolboxModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<ToolboxService>().As<IToolboxService>().SingleInstance();

            builder.RegisterType<ViewModels.ToolboxViewModel>()
                .AsSelf().As<IToolbox>().SingleInstance();
            builder.RegisterType<Views.ToolboxView>();
        }
    }
}
