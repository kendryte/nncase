namespace Nncase.TestFixture;

public interface IDataExtractor
{
    public List<string> GetFilesByOrdered(string dir);

    public Tensor[] GetTensors(string dir);

    public bool ShouldBeExtract(string dir);
}

public class RuntimeDataExtractor : IDataExtractor
{
    // FileFormat: (\d+)*[a-z]*
    public int GetDumpFileNum(string filePath)
    {
        var fileName = Path.GetFileName(filePath);
        var match = System.Text.RegularExpressions.Regex
            .Match(fileName, @"(\d+)*");
        return int.Parse(match.Groups[0].Value);
    }

    public int FileNumSorter(string x, string y)
    {
        var a = GetDumpFileNum(x);
        var b = GetDumpFileNum(y);
        return a.CompareTo(b);
    }

    public List<string> GetFilesByOrdered(string dir)
    {
        var fs = Directory.GetFiles(dir).ToList();
        fs.Sort(FileNumSorter);
        // remove out shape list
        fs.RemoveAt(0);
        return fs;
    }

    // todo: add selector
    // now only can be used for runtime
    public Tensor[] GetTensors(string dir)
    {
        var fs = GetFilesByOrdered(dir);
        return fs.Filter(ShouldBeExtract).Select(DataGenerator.FromTextFile).ToArray();
    }

    // used for transformer
    public bool ShouldBeExtract(string filePath)
    {
        var lower = Path.GetFileName(filePath).ToLower();
        return lower.Contains("mat") && lower.EndsWith("mul");
    }
}