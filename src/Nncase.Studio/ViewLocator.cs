using System;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Nncase.Studio.ViewModels;

namespace Nncase.Studio;

public class ViewLocator : IDataTemplate
{
    public Control Build(object data)
    {
        var name = data.GetType().FullName!.Replace("ViewModel", "View");
        var type = Type.GetType(name);

        // todo: what this??
        // var panel = new DockPanel();
        // var obj = (OptStr)data;
        // var l = new Label();
        // l.Content = obj.OptName;
        // var t = new TextBox();
        // t.Text = obj.Value;
        // t.Watermark = "default";
        // panel.Children.Add(l);
        // panel.Children.Add(t);
        // l.Width = 200;
        // t.Width = 400;
        // return panel;

        if (type != null)
        {
            return (Control)Activator.CreateInstance(type)!;
        }

        return new TextBlock { Text = "Not Found: " + name };
    }

    public bool Match(object data)
    {
        return data is ViewModelBase;
    }
}
