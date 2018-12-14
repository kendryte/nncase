using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;

namespace NnCase.Designer.Modules.MainWindow
{
    public class MainWindowModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<Views.MainWindowView>();
            builder.RegisterType<ViewModels.MainWindowViewModel>();
        }
    }
}
