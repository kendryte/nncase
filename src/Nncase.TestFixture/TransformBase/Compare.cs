using System.Diagnostics;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;
using static Nncase.TestFixture.TensorUtil;
using static Nncase.Utilities.DumpUtility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.TestFixture;

public static class TensorUtil
{
    public static int GetChannelAxis(Shape shape)
    {
        return GetChannelAxis(shape.ToValueArray());
    }

    public static int GetChannelAxis(int[] shape)
    {
        return Math.Max(0, 1 - (4 - shape.Length));
    }
    
    private static Tensor SliceTensor(Tensor tensor, int start, int length, int channelAxis = 1)
    {
        var s = tensor.ElementType.SizeInBytes;
        return Tensor.FromBytes(tensor.ElementType,
            tensor.BytesBuffer.Slice(start * s, length * s), tensor.Dimensions[(channelAxis + 1)..]);
    }
    
    public static (int, int) GetShapeInfo(int[] shape, int channelAxis = 1)
    {
        var i = channelAxis + 1;
        var channels = shape[..i].Aggregate(1, (a, b) => a * b);
        var size = shape[i..].Aggregate(1, (a, b) => a * b);
        return (channels, size);
    }
    
    public static Tensor[] SliceByChannel(Tensor tensor)
    {
        var channelAxis = GetChannelAxis(tensor.Shape);
        var (channels, size) = GetShapeInfo(tensor.Dimensions.ToArray(), channelAxis);
        return Enumerable.Range(0, channels).Select(i =>
                SliceTensor(tensor, size * i, size, channelAxis))
            .ToArray();
    }
}

public static class Comparator
{
    private static float Prod(float[] data1, float[] data2)
    {
        return data1.Zip(data2).Aggregate(0f, (f, tuple) => f + tuple.Item1 * tuple.Item2);
    }

    public static float[] CosSimilarity(IValue a, IValue b)
    {
        return CosSimilarity(a.AsTensors(), b.AsTensors());
    }

    public static float[] CosSimilarity(Tensor[] a, Tensor[] b)
    {
        return a.Zip(b).Select(CosSimilarity).ToArray();
    }

    public static float CosSimilarity((Tensor, Tensor) t)
    {
        return CosSimilarity(t.Item1, t.Item2);
    }

    public static float[][] CosSimilarity(IValue[] a, IValue[] b)
    {
        return a.Zip(b).Select(tuple => CosSimilarity(tuple.Item1, tuple.Item2)).ToArray();
    }

    public static float CosSimilarity(Tensor a, Tensor b)
    {
        var va = a.ToArray<float>();
        var vb = b.ToArray<float>();
        var v1 = Math.Sqrt(Prod(va, va));
        var v2 = Math.Sqrt(Prod(vb, vb));
        var sum = Prod(va, vb);
        return (float) (sum / (v1 * v2));
    }

    public static bool TensorValueCompare(TensorValue pre, TensorValue post, float thresh)
    {
        var v1 = pre.AsTensor();
        var v2 = post.AsTensor();
        Assert.Equal(v1.Shape, v2.Shape);
        Assert.Equal(v1.ElementType, v2.ElementType);
        var cosSim = CosSimilarity(v1, v2);
        Assert.True(cosSim > thresh, "cosSim:" + cosSim);
        return true;
    }

    public static bool TupleValueCompare(TupleValue a, TupleValue b, float thresh = 0.99f)
    {
        if (a.Count != b.Count)
        {
            return false;
        }

        foreach (var (t1, t2) in a.AsTensors().Zip(b.AsTensors()))
        {
            if (!TensorValueCompare(t1, t2, thresh))
            {
                return false;
            }
        }

        return true;
    }
    
    public static bool Compare(IValue pre, IValue post, float thresh = 0.99f) => (pre, post) switch
    {
        (TensorValue a, TensorValue b) => TensorValueCompare(a, b, thresh),
        (TupleValue a, TupleValue b) => TupleValueCompare(a, b, thresh),
        (_, _) => throw new ArgumentOutOfRangeException()
    };

    public static float[][] CompareByChannel(IValue pre, IValue post, int channelAxis = 1, float thresh = 0.99f) =>
        TensorsCompareByChannel(pre.AsTensors(), post.AsTensors(), channelAxis, thresh);

    public static float[] TensorCompareByChannel(Tensor pre, Tensor post, int channelAxis = 1, float thresh = 0.99f)
    {
        // todo:broadcast type???
        var v1 = SliceByChannel(pre);
        var v2 = SliceByChannel(post);
        // Debug.Assert(v1.Length == v2.Length);
        return v1.Zip(v2).Select(data => CosSimilarity(data.Item1, data.Item2)).ToArray();
    }

    public static float[][] TensorsCompareByChannel(Tensor[] pre, Tensor[] post, int channelAxis = 1,
        float thresh = 0.99f)
    {
        return pre.Zip(post).Select(tuple =>
                TensorCompareByChannel(tuple.Item1, tuple.Item2, channelAxis, thresh))
            .ToArray();
    }
}

public class LazyCompartor
{
    // count => Either<ErrorReason, CosSim>
    private Dictionary<int, LanguageExt.Either<string, float>> error = new();

    public bool Compare(IValue pre, IValue post, int count, float thresh = 0.99f)
    {
        if (pre is TensorValue a &&
            post is TensorValue b)
        {
            return Compare(a, b, thresh).Match(cos =>
                {
                    error[count] = cos;
                    return false;
                },
                () => true);
        }
        else
        {
            error[count] = $"pre is {pre.GetType().Name}, post is {post.GetType().Name}";
            return false;
        }
    }

    public void AddFailed(int count, string reason)
    {
        error[count] = reason;
    }
    
    public static Option<float> Compare(TensorValue pre, TensorValue post, float thresh)
    {
        var v1 = pre.AsTensor();
        var v2 = post.AsTensor();
        var cosSim = Comparator.CosSimilarity(v1, v2);
        if (cosSim < thresh)
        {
            return Option.Some(cosSim);
        }
        else
        {
            return Option.None;
        }
    }

    public void Run(Action f)
    {
        f();
        FailedAssert();
    }
    
    public void FailedAssert()
    {
        if (error.Count == 0)
        {
            return;
        }
        var errList = error.Select(v =>
        {
            var count = v.Key;
            var value = v.Value;
            var reason = value.Match(cos => $"CosSim:{cos}", s => s);
            return $"count:{count} error, reason: ${reason}";
        });
        var errInfo = string.Join("\n", errList);
        Assert.True(false, errInfo);
    }
}

public static class DetailComparator
{
    public static void DumpCompareDetailAnalysis(CompareResultByChannel[] resultByChannels, string path, int i)
    {
        var shape = resultByChannels.Length != 0
            ? SerializeShape(resultByChannels.Head().Shape)
            : "data all ok and not shape info";
        WriteResult(Path.Join(path, $"Analysis{i}"), resultByChannels, $"{shape}");
    }

    // for single file
    public static void CompareDetailAnalysis(IEnumerable<DetailCompareResult> data, float thresh = 0.99f)
    {
        var result = data.Select(dataByTensor => CompareDetailAnalysis(dataByTensor, thresh)).ToArray();
    }

    public static CompareResultByChannel[] CompareDetailAnalysis(DetailCompareResult dataByTensor, float thresh = 0.99f)
    {
         return dataByTensor.Enumerable().Where(result => result.IsOk(thresh)).ToArray();
    }

    public static IEnumerable<DetailCompareResult> CompareDetail(Tensor[] a, Tensor[] b)
    {
        Debug.Assert(a.Length == b.Length);
        return a.Zip(b).Select((t) =>
            CompareDetail(Value.FromTensor(t.Item1), Value.FromTensor(t.Item2), GetChannelAxis(t.Item1.Shape)));
    }
    
    public static DetailCompareResult CompareDetail(IValue a, IValue b, int channelAxis = 1)
    {
        var cos = Comparator.CompareByChannel(a, b, channelAxis);
        var LossInfo = CompareForAccuracyLoss(a, b);
        var size = a.AsTensors().Length;
        // todo: fix this
        return new DetailCompareResult(new []{new DetailCompareResultInfo(cos[0], LossInfo[0])});
    }

    public static AccuracyLossInfo[][] CompareForAccuracyLoss(IValue pre, IValue post) =>
        TensorsCompareForAccuracyLoss(pre.AsTensors(), post.AsTensors());

    public static AccuracyLossInfo[][] TensorsCompareForAccuracyLoss(Tensor[] pre, Tensor[] post) =>
        pre.Zip(post).Select(tuple => TensorCompareForAccuracyLoss(tuple.Item1, tuple.Item2)).ToArray();

    public static AccuracyLossInfo[] TensorCompareForAccuracyLoss(Tensor pre, Tensor post)
    {
        var v1 = pre.ToArray<float>();
        var v2 = post.ToArray<float>();
        var preShape = pre.Shape.ToValueArray();
        var index = new int[preShape.Length];
        return v1.Zip(v2).Select(data =>
        {
            index[^1] += 1;
            for (int i = preShape.Length - 1; i >= 0; i--)
            {
                if (index[i] > preShape[i])
                {
                    index[i] = 0;
                    index[i - 1] += 1;
                }
            }
            Console.WriteLine(string.Join(",", index.Select(x => x.ToString())));
            return new AccuracyLossInfo(data.Item1, data.Item2, index.ToArray(), pre, post);
        }).ToArray();
    }

    public static void DumpCompareDetail(DetailCompareResult compareResult, string resultRoot, int count)
    {
        // todo: fix this
        var (cosByChannel, LossInfo) = compareResult.infos.Head();
        // todo: insert separator for channel or other
        WriteResult(Path.Join(resultRoot, $"cos_{count}"), cosByChannel);

        using (var stream = new StreamWriter(Path.Join(resultRoot, count.ToString())))
        {
            var tensorShape = LossInfo[0].v1Tensor.Shape.ToValueArray();
            var (channels, size) = TensorUtil.GetShapeInfo(tensorShape, TensorUtil.GetChannelAxis(tensorShape));
            stream.WriteLine(SerializeShape(tensorShape));
            for (int i = 0; i < channels; i++)
            {
                stream.WriteLine(cosByChannel[i]);
                for (int j = 0; j < size; j++)
                {
                    stream.WriteLine(LossInfo[i * size + j]);
                }
            }
        }
    }

    public static void GenerateFullCompareInfo(Tensor[] a, Tensor[] b, string resultRoot)
    {
        GenerateFullCompareInfo(
            a.Select(x => (IValue) Value.FromTensor(x)).Zip(b.Select(x => (IValue) Value.FromTensor(x))), resultRoot);
    }

    public static void GenerateFullCompareInfo(IValue[] a, IValue[] b, string resultRoot)
    {
        if (a.Length != b.Length)
        {
            throw new InvalidOperationException($"tensor a and b should same length but a:{a.Length} b:{b.Length}");
        }
        GenerateFullCompareInfo(a.Zip(b), resultRoot);
    }
    
    public static void GenerateFullCompareInfo(IEnumerable<(IValue, IValue)> data, string resultRoot)
    {
        var counter = new Counter(1);
        foreach (var (originD, k230D) in data)
        {
            counter.Run(count => GenerateFullCompareInfo(resultRoot, originD, k230D, count));
        }
    }

    private static CompareResultByChannel[] GenerateFullCompareInfo(string resultRoot, IValue originD, IValue k230D,
        int count)
    {
        var result = DetailComparator.CompareDetail(originD, k230D);
        DetailComparator.DumpCompareDetail(result, resultRoot, count);
        var analysisResult = DetailComparator.CompareDetailAnalysis(result);
        DetailComparator.DumpCompareDetailAnalysis(analysisResult, resultRoot, count);
        return analysisResult;
    }
}

public record DetailCompareResultInfo(float[] CosList, AccuracyLossInfo[] AccuracyLossInfos)
{
    public int[] Shape => AccuracyLossInfos.Head().Shape;

    public IEnumerable<CompareResultByChannel> Enumerable()
    {
        var tensorShape = Shape;
        var (channels, size) = GetShapeInfo(tensorShape, GetChannelAxis(tensorShape));
        return System.Linq.Enumerable.Range(0, channels).Select(c => new CompareResultByChannel(CosList[c], AccuracyLossInfos[(c * size)..((c + 1) * size)]));
    }
}

public record DetailCompareResult(DetailCompareResultInfo[] infos)
{
    public bool IsTuple => infos.Length == 1;

    public IEnumerable<CompareResultByChannel> Enumerable()
    {
        return infos.Aggregate(System.Linq.Enumerable.Empty<CompareResultByChannel>(), 
            (channels, info) => channels.Concat(info.Enumerable()));
    }
}
    
public record CompareResultByChannel(float cos, AccuracyLossInfo[] LossInfo)
{
    public int[] Shape => LossInfo.Head().Shape;

    public bool IsOk(float thresh)
    {
        return cos < thresh || Losses.Length != 0;
    }

    // todo: more analysis strategy
    public AccuracyLossInfo[] Losses => LossInfo.Where(deviation =>
        (deviation.Ratio > 1.3 || deviation.Ratio < 0.7)).ToArray();

    public override string ToString()
    {
        var err = Losses;
        var percent = (float) err.Length / new Shape(Shape).Prod().FixedValue;
        return $"CompareResultByChannel Cos:{cos} \nLossCount/InputSize: {percent}\nLoss:\n{SerializeByColumn(err)}";
    }
}

public record AccuracyLossInfo(float v1, float v2, int[] index, Tensor v1Tensor, Tensor v2Tensor)
{
    public int[] Shape => v1Tensor.Shape.ToValueArray();
    public float Loss => Math.Abs(v1 - v2);
    // todo: v2 is 0.0000
    public float Ratio => (v1 == 0 && v2 == 0) || (Math.Abs(v1) < 1e-9 && v2 == 0) ? 1 : Math.Abs(v1 / v2);

    public override string ToString()
    {
        return
            $"AccuracyLossInfo v1 = {v1}, v2 = {v2}, index =[{string.Join(",", index.Select(x => x.ToString()))}], devi = {Loss}, ratio = {Ratio}";
    }
}