// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Newtonsoft.Json.Linq;
using Nncase;
using Nncase.IR;
using Nncase.IR.Tensors;
using Tuple = System.Tuple;

internal class SafetensorsEntry
{
    [JsonPropertyName("dtype")]
    public string DataType { get; init; }

    [JsonPropertyName("shape")]
    public long[] Shape { get; init; }

    [JsonPropertyName("data_offsets")]
    public long[] Offsets { get; init; }
}

internal static class HuggingFaceUtils
{
    public static Dictionary<string, object> GetConfigInfo(string path)
    {
        var config = new Dictionary<string, object>();
        if (File.Exists(path))
        {
            var configJson = File.ReadAllText(path);
            config = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, object>>(
                configJson);
            foreach (var key in config.Keys.ToList())
            {
                if (config[key] is JArray jArray)
                {
                    config[key] = string.Join(", ", jArray.Select(token => token.ToString()));
                }
            }
        }
        else
        {
            throw new FileNotFoundException(
                $"{config?["architectures"]}'s config.json not found in the specified directory.",
                path);
        }

        return config;
    }

    public static Dictionary<string, Tensor> GetAllWeights(string path)
    {
        var constTensors = new Dictionary<string, Tensor>();
        var constTensor = HuggingFaceUtils.LoadStateDict(path);
        foreach (var item in constTensor)
        {
            Console.WriteLine($"{item.Key}");
            if (item.Value is Tensor tensor)
            {
                constTensors.Add(item.Key, tensor.CastTo(DataTypes.Float32));
            }
        }

        return constTensors;
    }

    public static byte[] ReadBytes(this Stream stream, int count)
    {
        byte[] buffer = new byte[count];
        stream.Read(buffer, 0, count);
        return buffer;
    }

    internal static Dictionary<string, SafetensorsEntry> LoadIndex(Stream stream)
    {
        ulong uint64 = BitConverter.ToUInt64((ReadOnlySpan<byte>)stream.ReadBytes(8));
        if (uint64 > (ulong)int.MaxValue)
        {
            throw new ArgumentOutOfRangeException("length", "Length of JSON exceeded int.MaxValue, not supported yet");
        }

        return JsonSerializer.Deserialize<Dictionary<string, SafetensorsEntry>>(
                   Encoding.UTF8.GetString(stream.ReadBytes((int)uint64))) ??
               throw new NotImplementedException("Loaded header string failed to deserialize into the correct format.");
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
                !(keyValuePair.Key == "__metadata__")
                && (keysToKeep == null || keysToKeep.Contains(keyValuePair.Key)))
            {
                var datatype = ConvertToDataDType(keyValuePair.Value.DataType);

                // var tensor = new Tensor(datatype, new Shape(keyValuePair.Value.Shape));
                var shape = new Shape(keyValuePair.Value.Shape);
                if (
                    keyValuePair.Value.Offsets[1] - keyValuePair.Value.Offsets[0]
                    != datatype.SizeInBytes * shape.Size)
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
            }
        }

        throw new NotImplementedException("Unrecognized data type listed: " + dataType);
    }

    public static Tuple<List<double>, float> ComputeDefaultRopeParameters(
        Dictionary<string, object> config)
    {
        /*
         * base = config.rope_theta
           partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
           head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
           dim = int(head_dim * partial_rotary_factor)
         */
        var baseRoPETheta = (float)(double)config["rope_theta"];
        var partialRotaryFactor = 1.0; // config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0

        int headDim;

        if (config.TryGetValue("head_dim", out object headDimObj) && headDimObj is int headDim1)
        {
            headDim = headDim1;
        }
        else
        {
            int hiddenSize = (int)(long)config["hidden_size"];
            int numAttentionHeads = (int)(long)config["num_attention_heads"];
            headDim = hiddenSize / numAttentionHeads;
        }

        var dim = (int)(headDim * partialRotaryFactor);
        float attentionFactor = 1.0f;

        // Compute the inverse frequencies
        // 创建一个从 0 到 dim-1 的数组，步长为 2
        var arange = Enumerable
            .Range(0, dim)
            .Where(i => i % 2 == 0)
            .Select(i => (float)i)
            .ToArray();

        // 计算 inv_freq
        var inv_freq = arange
            .Select(i => 1.0 / Math.Pow(baseRoPETheta, i / dim))
            .ToArray()
            .ToList();
        return Tuple.Create(inv_freq, attentionFactor);
    }

    public class DynamicCache
    {
        public int SeenTokens;
        public List<object>? KeyCache;
        public List<object>? ValueCache;

        public int GetSeqLength(int layerCount = 0)
        {
            bool isEmptyLayer =
                KeyCache?.Count == 0
                || KeyCache?.Count <= layerCount
                || (int)KeyCache?[layerCount] == 0;
            var layer = (Call)KeyCache?[(Index)layerCount!];
            return isEmptyLayer ? 0 : layer.CheckedShape[-2].FixedValue;
        }

        public Tuple<Call, Call> Update(
            Call keyStates,
            Call valueStates,
            int layerCount,
            Dictionary<string, object> cacheKwargs)
        {
            if (layerCount == 0)
            {
                SeenTokens += (int)keyStates.CheckedShape[-2].FixedValue;
            }

            if (keyStates != null)
            {
                if (KeyCache.Count <= layerCount)
                {
                    for (int i = KeyCache.Count; i <= layerCount; i++)
                    {
                        KeyCache.Add(0);
                        ValueCache.Add(0);
                    }

                    // self.key_cache.append(key_states)
                    // self.value_cache.append(value_states)
                    KeyCache.Add(keyStates);
                    ValueCache.Add(valueStates);
                }
                else if ((int)KeyCache[layerCount] == 0)
                {
                    KeyCache[layerCount] = keyStates;
                    ValueCache[layerCount] = valueStates;
                }
                else
                {
                    KeyCache[layerCount] = Nncase.IR.F.Tensors.Concat(
                        new Nncase.IR.Tuple((Call)KeyCache[layerCount], keyStates),
                        -2);
                    ValueCache[layerCount] = Nncase.IR.F.Tensors.Concat(
                        new Nncase.IR.Tuple((Call)ValueCache[layerCount], valueStates),
                        -2);
                }
            }

            return Tuple.Create((Call)KeyCache[layerCount], (Call)ValueCache[layerCount]);
        }
    }
}
