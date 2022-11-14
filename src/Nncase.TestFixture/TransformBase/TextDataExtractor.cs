using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.TestFixture;

public class TextDataExtractor
{
    // FileNameFormat
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
    
    public IEnumerable<IGrouping<string, string>> GetFilesByGroup(string dir)
    {
        return GetFilesByOrdered(dir).GroupBy(file => string.Join(Separator, file.Split(Separator)[..2]));
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="dir"></param>
    /// <returns> dict:num$op_name -> num$op_name$param_name/returns>
    public Dictionary<string, IEnumerable<string>> GetFilesByOrderNum(string dir)
    {
        return GetFilesByGroup(dir)
            .ToDictionary(
            x =>
            {
                var split = x.Key.Split(Separator);
                return $"{split[0].Split(Path.DirectorySeparatorChar).Last()}{Separator}{split[1]}";
            },
            x => x.Select(s => s));
    }

    public IValue[] ExtractValues(string dir, Func<string, bool> Extractor)
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

    public IValue[] GetComputeResults(string dir) => ExtractValues(dir, IsResultFile);

    public IValue[] GetParams(string dir, int count) => ExtractValues(dir,
        file => IsParamFile(file) && GetCount(file) == count);

    public IValue[] GetValues(string dir)
    {
        return ExtractValues(dir, _ => true);
    }

    // used for transformer
    public bool DynamicMatmulOnlyExtract(string fileName)
    {
        var lower = fileName.ToLower();
        return lower.Contains("mat") && lower.EndsWith("mul");
    }

    public IValue[] OpExtract(string dir, string opName)
        => ExtractValues(dir, file => GetOpName(file) == opName);

    public Tensor[] MatmulExtract(string dir)
    {
        return ExtractValues(dir, DynamicMatmulOnlyExtract).Select(x => x.AsTensor()).ToArray();
    }
}