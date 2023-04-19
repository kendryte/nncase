using Microsoft.AspNetCore.Components.WebView.WindowsForms;
using Microsoft.Extensions.Hosting;

namespace Nncase.Studio.WinForms;

public partial class MainForm : Form
{
    public MainForm(IHost host)
    {
        InitializeComponent();

        blazorWebView.Services = host.Services;
        blazorWebView.RootComponents.Add<Main>("#app");
    }
}
