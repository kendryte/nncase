// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Linq;
using LanguageExt;
using Nncase.IR;
using Onnx;

namespace Nncase.Importer
{
    public sealed partial class OnnxImporter
    {
        private DataType GetDataType(string name)
        {
            return _graph.Input
                .Find(x => x.Name == name)
                .Match(
                    GetDataType,
                    () =>
                    {
                        if (_constTensors!.TryGetValue(name, out var tensor))
                        {
                            return GetDataType(tensor);
                        }
                        else
                        {
                            throw new InvalidDataException($"{name} DataType Info not found");
                        }
                    });
        }

        private DataType GetDataType(ValueInfoProto v)
        {
            return GetDataType(v.Type.TensorType.ElemType);
        }

        private DataType GetDataType(TensorProto tensor)
        {
            return GetDataType(tensor.DataType);
        }

        private DataType GetDataType(long onnxTypeIndex)
        {
            return GetDataType((int)onnxTypeIndex);
        }

        private DataType GetDataType(int onnxTypeIndex)
        {
            var dType = (TensorProto.Types.DataType)onnxTypeIndex;
            if (_typeMap.ContainsKey(dType))
            {
                return _typeMap[dType];
            }
            else
            {
                throw new InvalidDataException($"Not supported Datatype {dType}");
            }
        }
    }
}
