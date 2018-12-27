using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows;
using Autofac;
using NnCase.Designer.Modules.MainWindow.ViewModels;
using NnCase.Designer.Modules.MainWindow.Views;
using ReactiveUI;
using Splat;

namespace NnCase.Designer
{
    /// <summary>
    /// App.xaml 的交互逻辑
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
            var containerBuilder = new ContainerBuilder();
            RegisterAssemblies(containerBuilder);
            var resolver = new AutofacDependencyResolver(containerBuilder.Build());
            resolver.InitializeSplat();
            resolver.InitializeReactiveUI();
            Locator.Current = resolver;
            DispatcherUnhandledException += App_DispatcherUnhandledException;
        }

        private void App_DispatcherUnhandledException(object sender, System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
        {
            e.Handled = true;
            MessageBox.Show(e.Exception.ToString(), "NnCase", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        private void RegisterAssemblies(ContainerBuilder containerBuilder)
        {
            var assemblies = new List<Assembly>();
            assemblies.Add(typeof(App).Assembly);

            assemblies
                .AddShell()
                .AddInspector();

            containerBuilder.RegisterAssemblyModules(assemblies.ToArray());
        }

        [STAThread]
        public static void Main()
        {
            var app = new App();
            app.InitializeComponent();
            app.Run(new MainWindowView(Locator.Current.GetService<MainWindowViewModel>()));
        }
    }
}
