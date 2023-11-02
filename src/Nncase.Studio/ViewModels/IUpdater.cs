namespace Nncase.Studio.ViewModels;

public interface IUpdater
{
    public void UpdateCompileOption(CompileOptions options);

    public bool Validate();
}
