using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.IR;
using TorchSharp;

internal class SafetensorsEntry
{
    [JsonPropertyName("dtype")]
    public string DataType { get; init; }

    [JsonPropertyName("shape")]
    public long[] Shape { get; init; }

    [JsonPropertyName("data_offsets")]
    public long[] Offsets { get; init; }
}

static class HuggingFaceUtils
{


public static byte[] ReadBytes(this Stream stream, int count)
{
    byte[] buffer = new byte[count];
    stream.Read(buffer, 0, count);
    return buffer;
}

internal static Dictionary<string, SafetensorsEntry> LoadIndex(Stream stream)
    {
        ulong uint64 = BitConverter.ToUInt64((ReadOnlySpan<byte>) stream.ReadBytes(8));
        if (uint64 > (ulong) int.MaxValue)
            throw new ArgumentOutOfRangeException("length", "Length of JSON exceeded int.MaxValue, not supported yet");
        return JsonSerializer.Deserialize<Dictionary<string, SafetensorsEntry>>(Encoding.UTF8.GetString(stream.ReadBytes((int) uint64))) ?? throw new NotImplementedException("Loaded header string failed to deserialize into the correct format.");
    }

    public static Dictionary<string, Tensor> LoadStateDict(
        string path,
        List<string>? keysToKeep = null)
    {
        using (FileStream fileStream = File.OpenRead(path))
            return LoadStateDict((Stream) fileStream, keysToKeep: keysToKeep);
    }

    public static Dictionary<string, Tensor> LoadStateDict(
        Stream stream,
        bool leaveOpen = false,
        List<string>? keysToKeep = null)
    {
        Dictionary<string, SafetensorsEntry> dictionary1 = HuggingFaceUtils.LoadIndex(stream);
        long position = stream.Position;
        Dictionary<string, Tensor> dictionary2 = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, SafetensorsEntry> keyValuePair in dictionary1)
        {
            if (!(keyValuePair.Key == "__metadata__") && (keysToKeep == null || keysToKeep.Contains(keyValuePair.Key)))
            {
                var datatype = ConvertToDataDType(keyValuePair.Value.DataType);
                // var tensor = new Tensor(datatype, new Shape(keyValuePair.Value.Shape));
                var shape = new Shape(keyValuePair.Value.Shape);
                if (keyValuePair.Value.Offsets[1] - keyValuePair.Value.Offsets[0] != datatype.SizeInBytes * shape.Size)
                    throw new NotImplementedException("Error when loading tensor " + keyValuePair.Key + " - mismatched # of elements");
                stream.Position = position + keyValuePair.Value.Offsets[0];
                var tensor = Tensor.FromStream(datatype, stream, shape);
                dictionary2.Add(keyValuePair.Key, tensor);
            }
        }
        if (!leaveOpen)
            stream.Close();
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
                      return DataTypes.Int8;
                    break;
                  case 'U':
                    if (dataType == "U8")
                      return DataTypes.UInt8;
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
                      return DataTypes.BFloat16;
                    break;
                  case 'O':
                    if (dataType == "BOOL")
                      return DataTypes.Boolean;
                    break;
                }
                break;
            }
          }
          throw new NotImplementedException("Unrecognized data type listed: " + dataType);
        }
}
