namespace Nncase.TestFixture;

public class TextDataExtractor
{
    // FileFormat
    // input: (\d+)*$[a-z]*
    // param: (\d+)*$[a-z]*$[a-z]*
    public int GetDumpFileNum(string filePath)
    {
        var fileName = Path.GetFileName(filePath);
        if (fileName.Contains("out_shape_list"))
        {
            return -1;
        }
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

    // now only can be used for runtime, evaluator dump format not be modified
    public Tensor[] ExtractTensors(string dir, Func<string, bool> Extractor)
    {
        var fs = GetFilesByOrdered(dir);
        return fs
            .Filter(filePath => Extractor(Path.GetFileName(filePath)))
            .Select(DataGenerator.FromTextFile)
            .ToArray();
    }


    public static char Separator => '$';
    public int GetCount(string file) => int.Parse(file.Split(Separator).Head());

    public string GetOpName(string file) => file.Split(Separator)[1];

    // todo: is param
    public string GetParamName(string file) => file.Split(Separator).Last();

    public bool IsResultFile(string file) => file.Count(c => c == Separator) == 1;
    public bool IsParamFile(string file) => file.Count(c => c == Separator) == 2;

    public Tensor[] GetComputeResults(string dir) => ExtractTensors(dir, IsResultFile);

    public Tensor[] GetParams(string dir, int count) => ExtractTensors(dir,
        file => IsParamFile(file) && GetCount(file) == count);

    public Tensor[] GetTensors(string dir)
    {
        return ExtractTensors(dir, _ => true);
    }

    // used for transformer
    public bool DynamicMatmulOnlyExtract(string fileName)
    {
        var lower = fileName.ToLower();
        return lower.Contains("mat") && lower.EndsWith("mul");
    }

    public Tensor[] OpExtract(string dir, string opName)
        => ExtractTensors(dir, file => GetOpName(file) == opName);

    public Tensor[] MatmulExtract(string dir)
    {
        return ExtractTensors(dir, DynamicMatmulOnlyExtract);
    }
}