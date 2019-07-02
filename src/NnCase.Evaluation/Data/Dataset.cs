using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;

namespace NnCase.Evaluation.Data
{
    public abstract class Dataset
    {
        private readonly Shape _inputShape;
        private readonly List<string> _fileNames = new List<string>();

        public int BatchSize => _inputShape[0];

        public Dataset(string path, Func<string, bool> fileFilter, Shape inputShape, float min, float std)
        {
            _inputShape = inputShape;

            if (Directory.Exists(path))
            {
                foreach (var fileName in Directory.EnumerateFiles(path, "*", new EnumerationOptions { RecurseSubdirectories = true }))
                {
                    if (fileFilter(fileName))
                        _fileNames.Add(fileName);
                }
            }
            else if (File.Exists(path))
            {
                if (fileFilter(path))
                    _fileNames.Add(path);
            }

            if (_fileNames.Count == 0)
                throw new ArgumentException("Invalid dataset, should contain one file at least");

            var samples = _fileNames.Count / BatchSize * BatchSize;
            if (samples != _fileNames.Count)
                _fileNames.RemoveRange(samples, _fileNames.Count - samples);
        }

        public async IAsyncEnumerable<DenseTensor<float>> GetBatchesAsync()
        {
            var sliceShape = _inputShape.Clone();
            sliceShape[0] = 1;

            for (int batch = 0; batch < _fileNames.Count / BatchSize; batch++)
            {
                var tensor = new DenseTensor<float>(ShapeUtility.ComputeSize(_inputShape));
                var sliceSize = (int)(tensor.Length / BatchSize);

                for (int i = 0; i < BatchSize; i++)
                {
                    var slice = tensor.Buffer.Slice(i * sliceSize, sliceSize);
                    var file = await File.ReadAllBytesAsync(_fileNames[batch * BatchSize + i]);
                    Process(file, slice.Span, sliceShape);
                }

                yield return tensor;
            }
        }

        protected abstract void Process(ReadOnlySpan<byte> src, Span<float> dest, Shape shape);
    }
}
