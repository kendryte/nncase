using System;
using System.IO;
using System.Linq;
using LanguageExt;
using Onnx;
using static Onnx.AttributeProto.Types;

namespace Nncase.Importer
{
    public sealed partial class OnnxImporter
    {
        // todo:refactor, remove x=>x by add extension and replace if with ?
        Option<T> GetAttr<T>(NodeProto n, string attr, AttributeType type, Func<AttributeProto, T> func)
        {
            return n.Attribute
                .Find(x => x.Name == attr)
                .Map(
                    x =>
                    {
                        if (x.Type == type)
                        {
                            return func(x);
                        }
                        else
                        {
                            throw new InvalidDataException(
                                $"Find {attr} but type is mismatch, expect float, buf {x.Type}");
                        }
                    });
        }

        T GetAttrSafe<T>(NodeProto n, string attr, AttributeType type,
            Func<AttributeProto, T> func, T defaultValue)
        {
            return GetAttr(n, attr, type, func).Match(x => x, () => defaultValue);
        }

        Option<T> GetAttrOption<T>(NodeProto n, string attr, AttributeType type,
            Func<AttributeProto, T> func)
        {
            return GetAttr(n, attr, type, func);
        }
        
        T GetAttrUnSafe<T>(NodeProto n, string attr, AttributeType type,
            Func<AttributeProto, T> func)
        {
            return GetAttr(n, attr, type, func)
                .Match(
                    x => x,
                    () => throw new InvalidDataException($"Cannot find node attr {attr} in node {n}"));
        }
        
        long GetIntAttribute(NodeProto n, string attr)
        {
            return GetAttrUnSafe(n, attr, AttributeType.Int, x => x.I);
        }
        
        long GetIntAttribute(NodeProto n, string attr, long defaultValue)
        {
            return GetAttrSafe(n, attr, AttributeType.Int, x => x.I, defaultValue);
        }
        
        float GetFloatAttribute(NodeProto n, string attr, float defaultValue)
        {
            return GetAttrSafe(n, attr, AttributeType.Float, x => x.F, defaultValue);
        }

        bool GetBoolAttribute(NodeProto n, string attr, bool defaultValue)
        {
            return GetIntAttribute(n, attr, defaultValue ? 1 : 0) != 0;
        }

        long[] GetAxisAttribute(NodeProto n, string attr)
        {
            return GetIntsAttribute(n, attr);
        }

        long[] GetIntsAttribute(NodeProto n, string attr)
        {
            return GetAttrUnSafe(n, attr, AttributeType.Ints, x => x.Ints.ToArray());
        }
        
        Option<long[]> GetOptionIntsAttribute(NodeProto n, string attr)
        {
            return GetAttrOption(n, attr, AttributeType.Ints, x => x.Ints.ToArray());
        }
        
        long[] GetIntsAttribute(NodeProto n, string attr, int[] defaultValue)
        {
            return GetAttrSafe(n, attr, AttributeType.Ints, x => x.Ints.ToArray(),
                defaultValue.Select(x => (long)x).ToArray());
        }

        long[] GetIntsAttribute(NodeProto n, string attr, int defaultValue, int count)
        {
            return GetAttrSafe(n, attr, AttributeType.Ints, x => x.Ints.ToArray(), 
                Enumerable.Repeat<long>(defaultValue, count).ToArray());
        }
        
        string GetStringAttribute(NodeProto n, string attr, string defaultValue)
        {
            return GetAttrSafe(n, attr, AttributeType.String, x => x.S.ToString(), defaultValue);
        }
        
        string GetStringAttribute(NodeProto n, string attr)
        {
            return GetAttrUnSafe(n, attr, AttributeType.String, x => x.S.ToString());
        }

        TensorProto GetTensorProtoAttribute(NodeProto n, string attr)
        {
            return GetAttrUnSafe(n, attr, AttributeType.Tensor, x => x.T);
        }
    }
}