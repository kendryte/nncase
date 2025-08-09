// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using NetFabric.Hyperlinq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Nncase;
using Nncase.IR;
using Nncase.IR.NN;

using Tuple = System.Tuple;

public class MyJsonConverter
{
    public static Dictionary<string, object> ParseNestedJson(string json)
    {
        var root = JsonConvert.DeserializeObject<Dictionary<string, object>>(
            json,
            new JsonSerializerSettings
            {
                DateParseHandling = DateParseHandling.None,
            });

        ProcessDictionary(root);
        return root;
    }

    private static void ProcessDictionary(IDictionary<string, object> dict)
    {
        foreach (var key in dict.Keys.ToList())
        {
            var value = dict[key];

            if (value is JObject jObject)
            {
                var subDict = jObject.ToObject<Dictionary<string, object>>();
                ProcessDictionary(subDict);
                dict[key] = subDict;
            }
            else if (value is JArray jArray)
            {
                var list = ProcessArray(jArray);
                dict[key] = list;
            }
            else if (value is JValue jValue)
            {
                dict[key] = jValue.Value;
            }
        }
    }

    private static List<object> ProcessArray(JArray jArray)
    {
        var list = new List<object>();
        foreach (var item in jArray)
        {
            switch (item.Type)
            {
                case JTokenType.Object:
                    var subDict = ((JObject)item).ToObject<Dictionary<string, object>>();
                    ProcessDictionary(subDict);
                    list.Add(subDict);
                    break;
                case JTokenType.Array:
                    list.Add(ProcessArray((JArray)item));
                    break;
                default:
                    list.Add(((JValue)item).Value);
                    break;
            }
        }

        return list;
    }
}

internal static class HuggingFaceUtils
{
    public static T GetNestedValue<T>(this Dictionary<string, object> dict, params object[] keys)
    {
        object current = dict;
        var keyPath = new List<object>();

        foreach (var key in keys)
        {
            keyPath.Add(key);

            switch (current)
            {
                case Dictionary<string, object> d:
                    var keyString = key.ToString() ?? throw new ArgumentException("Key cannot be null");
                    if (!d.TryGetValue(keyString, out current!))
                    {
                        Console.WriteLine($"Key not found: {key}");
                        Console.WriteLine($"Current key path: {string.Join(" -> ", keyPath)}");
                        Console.WriteLine($"Current dictionary: {JsonConvert.SerializeObject(d, Formatting.Indented)}");
                        throw new KeyNotFoundException($"Key '{key}' not found in dictionary. Path: {string.Join(" -> ", keyPath)}");
                    }

                    break;

                case List<object> l when key is int index:
                    if (index < 0 || index >= l.Count)
                    {
                        Console.WriteLine($"Index out of range: {index}");
                        Console.WriteLine($"Current key path: {string.Join(" -> ", keyPath)}");
                        Console.WriteLine($"Current list: {JsonConvert.SerializeObject(l, Formatting.Indented)}");
                        throw new ArgumentOutOfRangeException(nameof(dict), $"Index {index} is out of range for the list. Path: {string.Join(" -> ", keyPath)}");
                    }

                    current = l[index];
                    break;

                default:
                    Console.WriteLine($"Invalid operation at key: {key}");
                    Console.WriteLine($"Current key path: {string.Join(" -> ", keyPath)}");
                    Console.WriteLine($"Current object type: {current?.GetType().Name ?? "null"}");
                    throw new InvalidOperationException($"Invalid operation at key '{key}'. Path: {string.Join(" -> ", keyPath)}");
            }
        }

        return (T)current;
    }

    public static Dictionary<string, object> GetConfigInfo(string path)
    {
        _ = new Dictionary<string, object>();
        Dictionary<string, object>? config;
        if (File.Exists(path))
        {
            var configJson = File.ReadAllText(path);
            config = MyJsonConverter.ParseNestedJson(configJson);
        }
        else
        {
            throw new FileNotFoundException(
                "config.json not found in the specified directory.",
                path);
        }

        return config;
    }

    public static Dictionary<string, Tensor> LoadAllTensorsFromFile(string path)
    {
        var constTensors = new Dictionary<string, Tensor>();
        var constTensor = HuggingFaceUtils.LoadStateDict(path);
        foreach (var item in constTensor)
        {
            if (item.Value is Tensor tensor)
            {
                constTensors.Add(item.Key, tensor.CastTo(DataTypes.Float32));
            }
        }

        return constTensors;
    }

    public static Dictionary<string, string> LoadWeightToFileMap(string modelDir)
    {
        var indexJsonPath = Path.Combine(modelDir, "model.safetensors.index.json");
        var singleModelPath = Path.Combine(modelDir, "model.safetensors");

        // Case 1: Large model with index file
        if (File.Exists(indexJsonPath))
        {
            string json = File.ReadAllText(indexJsonPath);
            using var doc = JsonDocument.Parse(json);

            var map = new Dictionary<string, string>();
            if (doc.RootElement.TryGetProperty("weight_map", out var weightMap))
            {
                foreach (var prop in weightMap.EnumerateObject())
                {
                    var stringValue = prop.Value.GetString();
                    if (stringValue != null)
                    {
                        map[prop.Name] = stringValue;
                    }
                }
            }
            else
            {
                Console.WriteLine("weight_map not found in index.json");
            }

            return map;
        }

        // Case 2: Small model with single safetensors file
        else if (File.Exists(singleModelPath))
        {
            // Load all tensor names from the single file and map them to the same file
            var tensors = LoadStateDict(singleModelPath);
            var map = new Dictionary<string, string>();

            foreach (var tensorName in tensors.Keys)
            {
                if (tensorName != "__metadata__")
                {
                    map[tensorName] = "model.safetensors";
                }
            }

            return map;
        }
        else
        {
            throw new FileNotFoundException($"Neither index file ({indexJsonPath}) nor single model file ({singleModelPath}) found in directory: {modelDir}");
        }
    }

    public static byte[] ReadBytes(this Stream stream, int count)
    {
        byte[] buffer = new byte[count];
        stream.Read(buffer, 0, count);
        return buffer;
    }

    public static Dictionary<string, Tensor> LoadStateDict(
        string path,
        List<string>? keysToKeep = null)
    {
        using (FileStream fileStream = File.OpenRead(path))
        {
            return LoadStateDict((Stream)fileStream, keysToKeep: keysToKeep);
        }
    }

    public static Dictionary<string, Tensor> LoadStateDict(
        Stream stream,
        bool leaveOpen = false,
        List<string>? keysToKeep = null)
    {
        Dictionary<string, SafetensorsEntry> dictionary1 = HuggingFaceUtils.LoadIndex(stream);
        long position = stream.Position;
        var dictionary2 = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, SafetensorsEntry> keyValuePair in dictionary1)
        {
            if (
                keyValuePair.Key != "__metadata__"
                && (keysToKeep == null || keysToKeep.Contains(keyValuePair.Key)))
            {
                var dataTypeString = keyValuePair.Value.DataType ?? throw new InvalidOperationException($"DataType is null for key {keyValuePair.Key}");
                var datatype = ConvertToDataDType(dataTypeString);

                var shapeArray = keyValuePair.Value.Shape ?? throw new InvalidOperationException($"Shape is null for key {keyValuePair.Key}");
                var shape = new RankedShape(shapeArray);

                var offsets = keyValuePair.Value.Offsets ?? throw new InvalidOperationException($"Offsets is null for key {keyValuePair.Key}");
                if (offsets[1] - offsets[0] != datatype.SizeInBytes * shape.Size)
                {
                    throw new NotImplementedException(
                        "Error when loading tensor "
                            + keyValuePair.Key
                            + " - mismatched # of elements");
                }

                stream.Position = position + keyValuePair.Value.Offsets[0];
                var tensor = Tensor.FromStream(datatype, stream, shape);
                dictionary2.Add(keyValuePair.Key, tensor);
            }
        }

        if (!leaveOpen)
        {
            stream.Close();
        }

        return dictionary2;
    }

    internal static Dictionary<string, SafetensorsEntry> LoadIndex(Stream stream)
    {
        ulong uint64 = BitConverter.ToUInt64((ReadOnlySpan<byte>)stream.ReadBytes(8));
        if (uint64 > (ulong)int.MaxValue)
        {
            throw new ArgumentOutOfRangeException("stream", "Length of JSON exceeded int.MaxValue, not supported yet");
        }

        return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, SafetensorsEntry>>(
                   Encoding.UTF8.GetString(stream.ReadBytes((int)uint64))) ??
               throw new NotImplementedException("Loaded header string failed to deserialize into the correct format.");
    }

    private static DataType ConvertToDataDType(string dataType)
    {
        if (dataType != null)
        {
            switch (dataType.Length)
            {
                case 2:
                    switch (dataType[0])
                    {
                        case 'I':
                            if (dataType == "I8")
                            {
                                return DataTypes.Int8;
                            }

                            break;
                        case 'U':
                            if (dataType == "U8")
                            {
                                return DataTypes.UInt8;
                            }

                            break;
                    }

                    break;
                case 3:
                    switch (dataType[1])
                    {
                        case '1':
                            switch (dataType)
                            {
                                case "F16":
                                    return DataTypes.Float16;
                                case "I16":
                                    return DataTypes.Int16;
                            }

                            break;
                        case '3':
                            switch (dataType)
                            {
                                case "F32":
                                    return DataTypes.Float32;
                                case "I32":
                                    return DataTypes.Int32;
                            }

                            break;
                        case '6':
                            switch (dataType)
                            {
                                case "F64":
                                    return DataTypes.Float64;
                                case "I64":
                                    return DataTypes.Int64;
                            }

                            break;
                    }

                    break;
                case 4:
                    switch (dataType[1])
                    {
                        case 'F':
                            if (dataType == "BF16")
                            {
                                return DataTypes.BFloat16;
                            }

                            break;
                        case 'O':
                            if (dataType == "BOOL")
                            {
                                return DataTypes.Boolean;
                            }

                            break;
                    }

                    break;
                case 7:
                    switch (dataType[4])
                    {
                        case '4':
                            if (dataType == "F8_E4M3")
                            {
                                return DataTypes.Float8E4M3;
                            }

                            break;
                        case '5':
                            if (dataType == "F8_E5M2")
                            {
                                return DataTypes.Float8E5M2;
                            }

                            break;
                    }

                    break;
            }
        }

        throw new NotImplementedException("Unrecognized data type listed: " + dataType);
    }

    // public class DynamicCache
    // {
    //     public int SeenTokens;
    //     public List<object>? KeyCache;
    //     public List<object>? ValueCache;
    //     public long GetSeqLength(int layerCount = 0)
    //     {
    //         bool isEmptyLayer =
    //             KeyCache?.Count == 0
    //             || KeyCache?.Count <= layerCount
    //             || (int)KeyCache?[layerCount] == 0;
    //         var layer = (Call)KeyCache?[(Index)layerCount!];
    //         return isEmptyLayer ? 0 : layer.CheckedShape[-2].FixedValue;
    //     }
    //     public Tuple<Call, Call> Update(
    //         Call keyStates,
    //         Call valueStates,
    //         int layerCount,
    //         Dictionary<string, object> cacheKwargs)
    //     {
    //         if (layerCount == 0)
    //         {
    //             SeenTokens += (int)keyStates.CheckedShape[-2].FixedValue;
    //         }
    //         if (keyStates != null)
    //         {
    //             if (KeyCache.Count <= layerCount)
    //             {
    //                 for (int i = KeyCache.Count; i <= layerCount; i++)
    //                 {
    //                     KeyCache.Add(0);
    //                     ValueCache.Add(0);
    //                 }
    //                 // self.key_cache.append(key_states)
    //                 // self.value_cache.append(value_states)
    //                 KeyCache.Add(keyStates);
    //                 ValueCache.Add(valueStates);
    //             }
    //             else if ((int)KeyCache[layerCount] == 0)
    //             {
    //                 KeyCache[layerCount] = keyStates;
    //                 ValueCache[layerCount] = valueStates;
    //             }
    //             else
    //             {
    //                 KeyCache[layerCount] = Nncase.IR.F.Tensors.Concat(
    //                     new Nncase.IR.Tuple((Call)KeyCache[layerCount], keyStates),
    //                     -2);
    //                 ValueCache[layerCount] = Nncase.IR.F.Tensors.Concat(
    //                     new Nncase.IR.Tuple((Call)ValueCache[layerCount], valueStates),
    //                     -2);
    //             }
    //         }
    //         return Tuple.Create((Call)KeyCache[layerCount], (Call)ValueCache[layerCount]);
    //     }
    // }
}

internal static class ModelUtils
{
    /// <summary>
    /// huggingface utils functions: compute rope args.
    /// </summary>
    /// <param name="config">Get [rope_theta, head_dim, num_attention_heads, hidden_size].</param>
    /// <returns>repo parameters.</returns>
    public static Tuple<float[], float> ComputeDefaultRopeParameters(
        Dictionary<string, object> config)
    {
        /*
         * base = config.rope_theta
           partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
           head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
           dim = int(head_dim * partial_rotary_factor)
         */
        var baseRoPETheta = (float)Convert.ToDouble(config["rope_theta"]);
        var partialRotaryFactor = 1.0; // config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        if (config.ContainsKey("partial_rotary_factor"))
        {
            partialRotaryFactor = (float)Convert.ToDouble(config["partial_rotary_factor"]);
        }

        long headDim;

        if (config.TryGetValue("head_dim", out var headDimObj))
        {
            headDim = (long)headDimObj;
        }
        else
        {
            long hiddenSize = (long)config["hidden_size"];
            long numAttentionHeads = (long)config["num_attention_heads"];
            headDim = hiddenSize / numAttentionHeads;
        }

        var dim = (int)(headDim * partialRotaryFactor);
        float attentionFactor = 1.0f;

        // Compute the inverse frequencies
        var arange = Enumerable
            .Range(0, dim)
            .Where(i => i % 2 == 0)
            .Select(i => (float)i)
            .ToArray();

        // Compute inv_freq
        var inv_freq = arange
            .Select(i => 1f / MathF.Pow(baseRoPETheta, i / dim))
            .ToArray();
        return Tuple.Create(inv_freq, attentionFactor);
    }

    /// <summary>
    /// huggingface utils functions: compute rope args.
    /// </summary>
    /// <param name="config">Get [rope_theta, head_dim, num_attention_heads, hidden_size].</param>
    /// <returns>repo parameters.</returns>
    public static Tuple<float[], float> ComputeLlama3RopeParameters(
        Dictionary<string, object> config)
    {
        /*
         * base = config.rope_theta
           partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
           head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
           dim = int(head_dim * partial_rotary_factor)
         */

        var (invFreq, attentionFactor) = ComputeDefaultRopeParameters(config);
        var factor = (float)config.GetNestedValue<double>("rope_scaling", "factor");
        var lowFreqFactor = (float)config.GetNestedValue<double>("rope_scaling", "low_freq_factor");
        var highFreqFactor = (float)config.GetNestedValue<double>("rope_scaling", "high_freq_factor");
        var oldContextLen = config.GetNestedValue<long>("rope_scaling", "original_max_position_embeddings");

        var lowFreqWavelen = oldContextLen / lowFreqFactor;
        var highFreqWavelen = oldContextLen / highFreqFactor;

        var waveLen = invFreq.Select(f => 2 * MathF.PI / f).ToList();

        var invFreqLlama = invFreq.Zip(waveLen, (f, w) => w > lowFreqWavelen ? f / factor : f).ToArray();

        var smoothFactor = waveLen.Select(w =>
        {
            var denominator = highFreqFactor - lowFreqFactor;
            return denominator != 0 ? ((oldContextLen / w) - lowFreqFactor) / denominator : 0;
        }).ToArray();

        var smoothedInvFreq = smoothFactor.Zip(invFreqLlama, (s, f) => ((1 - s) * (f / factor)) + (s * f)).ToArray();

        var isMediumFreq = waveLen.Select((w, i) => !(w < highFreqWavelen) && !(w > lowFreqWavelen)).ToArray();

        invFreqLlama = invFreqLlama.Zip(isMediumFreq, (f, isMed) => isMed ? smoothedInvFreq[invFreqLlama.IndexOf(f)] : f).ToArray();

        return Tuple.Create(invFreqLlama, attentionFactor);
    }

    public static Tuple<float[], float> RoPEInit(Dictionary<string, object> config)
    {
        ArgumentNullException.ThrowIfNull(config);

        string type = "default";
        if (config.ContainsKey("rope_type"))
        {
            type = config.GetNestedValue<string>("rope_type");
        }
        else if (config.TryGetValue("rope_scaling", out var ropeScaling) && ropeScaling is not null)
        {
            type = config.GetNestedValue<string>("rope_scaling", "rope_type");
        }

        return type switch
        {
            "default" => ModelUtils.ComputeDefaultRopeParameters(config),
            "llama3" => ModelUtils.ComputeLlama3RopeParameters(config),
            _ => throw new NotImplementedException($"RoPE function {type} need to impl"),
        };
    }

    public static Call ActFunc(Call data, string actType)
    {
        switch (actType)
        {
            case "silu":
                data = Nncase.IR.F.NN.Sigmoid(data) * data;
                break;
            default:
                throw new ArgumentException("LLM act type not support!");
        }

        return data;
    }

    public static (int[] Lanes, int[] Axes) GetQKVVectorizeParams(IPagedAttentionConfig config, AttentionDimKind[] qLayout)
    {
        var lanes = new List<int>();
        var axes = new List<int>();
        for (int i = 0; i < config.VectorizedAxes.Count; i++)
        {
            if (config.VectorizedAxes[i] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.NumKVHeads)
            {
                axes.Add(config.VectorizedAxes[i] switch
                {
                    PagedKVCacheDimKind.NumKVHeads => qLayout.IndexOf(AttentionDimKind.Head),
                    PagedKVCacheDimKind.HeadDim => qLayout.IndexOf(AttentionDimKind.Dim),
                    _ => throw new ArgumentOutOfRangeException(nameof(config)),
                });
                lanes.Add(config.Lanes[i]);
            }
        }

        return (lanes.ToArray(), axes.ToArray());
    }

    public static int[] GetLayoutPerm(AttentionDimKind[] inputLayout, AttentionDimKind[] targetLayout)
    {
        return targetLayout.Select(i => inputLayout.IndexOf(i)).ToArray();
    }
}

internal class SafetensorsEntry
{
    [JsonPropertyName("dtype")]
    public string? DataType { get; init; }

    [JsonPropertyName("shape")]
    public long[]? Shape { get; init; }

    [JsonPropertyName("data_offsets")]
    public long[]? Offsets { get; init; }
}
