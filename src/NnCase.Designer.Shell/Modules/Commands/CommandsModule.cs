using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;
using NnCase.Designer.Commands;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.Commands
{
    public class CommandsModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<CommandRouter>().As<ICommandRouter>().SingleInstance();
            builder.RegisterType<CommandService>().As<ICommandService>().SingleInstance();
        }
    }
}
