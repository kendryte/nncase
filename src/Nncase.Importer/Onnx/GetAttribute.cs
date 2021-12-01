using System.IO;
using LanguageExt;
using Onnx;

namespace Nncase.Importer
{
    public sealed partial class OnnxImporter
    {
        Option<AttributeProto> FindAttr(NodeProto n, string attr)
        {
            return n.Attribute.Find(x => x.Name == attr);
        }
        
        float GetFloatAttribute(NodeProto n, string attr)
        {
            return FindAttr(n, attr)
                .Match(
                    x =>
                    {
                        if (x.Type == AttributeProto.Types.AttributeType.Float)
                        {
                            return x.F;
                        }
                        else
                        {
                            throw new InvalidDataException($"Find {attr} but type is mismatch, expect float, buf {x.Type}");
                        }
                    },
                    () => throw new InvalidDataException($"Cannot find node attr {attr} in node {n}"));
        }
    }
}