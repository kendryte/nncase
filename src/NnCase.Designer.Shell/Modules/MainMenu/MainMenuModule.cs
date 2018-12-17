using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;
using NnCase.Designer.Services;

namespace NnCase.Designer.Modules.MainMenu
{
    public class MainMenuModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<MenuBuilder>()
                .As<IMenuBuilder>();

            builder.RegisterInstance(MenuDefinictions.MainMenuBar)
                .PreserveExistingDefaults();
            builder.RegisterInstance(MenuDefinictions.FileMenu)
                .PreserveExistingDefaults();
            builder.RegisterInstance(MenuDefinictions.FileNewOpenMenuGroup)
                .PreserveExistingDefaults();

            builder.RegisterType<Views.MainMenuView>();
            builder.RegisterType<ViewModels.MainMenuViewModel>()
                .AsSelf().As<IMenu>().SingleInstance();
        }
    }
}
