using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.Shell
{
    public class ShellModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<Views.ShellView>();
            builder.RegisterType<ViewModels.ShellViewModel>()
                .AsSelf().As<IShell>().SingleInstance();
        }
    }
}
