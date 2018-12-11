using Avalonia;
using Avalonia.Markup.Xaml;

namespace NnCase.Converter.Shell
{
    public class App : Application
    {
        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
