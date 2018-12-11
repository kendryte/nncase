using System;
using Avalonia;
using Avalonia.Logging.Serilog;
using NnCase.Converter.Shell.ViewModels;
using NnCase.Converter.Shell.Views;

namespace NnCase.Converter.Shell
{
    class Program
    {
        static void Main(string[] args)
        {
            BuildAvaloniaApp().Start<MainWindow>(() => new MainWindowViewModel());
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .UseReactiveUI()
                .LogToDebug();
    }
}
