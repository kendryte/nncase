// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

public class QuantizedWeightsInfo
{
    public static readonly QuantizedWeightsInfo Default;
    private readonly Tensor _scales;
    private readonly Tensor? _biases;
    private readonly long _weightGroupSizeH;
    private readonly long _weightGroupSizeW;

    static QuantizedWeightsInfo()
    {
        Default = new QuantizedWeightsInfo();
    }

    public QuantizedWeightsInfo()
    {
        _scales = Tensor.Zero(DataTypes.Float32);
        _biases = Tensor.Zero(DataTypes.Float32);
        _weightGroupSizeH = 0;
        _weightGroupSizeW = 0;
    }

    public QuantizedWeightsInfo(Tensor scales, Tensor biases, long weightGroupSizeH, long weightGroupSizeW)
    {
        if (scales == null)
        {
            throw new ArgumentNullException(nameof(scales));
        }

        if (biases == null)
        {
            throw new ArgumentNullException(nameof(biases), "Biases matrix cannot be null for this constructor. Use the constructor without biases if biases are not applicable.");
        }

        if (weightGroupSizeH <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weightGroupSizeH), "Weight group size H (dimension 0) must be positive.");
        }

        if (weightGroupSizeW <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weightGroupSizeW), "Weight group size W (dimension 1) must be positive.");
        }

        _scales = scales;
        _biases = biases;
        _weightGroupSizeH = weightGroupSizeH;
        _weightGroupSizeW = weightGroupSizeW;
    }

    public QuantizedWeightsInfo(Tensor scales, long weightGroupSizeH, long weightGroupSizeW)
    {
        if (scales == null)
        {
            throw new ArgumentNullException(nameof(scales));
        }

        if (weightGroupSizeH <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weightGroupSizeH), "Weight group size H (dimension 0) must be positive.");
        }

        if (weightGroupSizeW <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weightGroupSizeW), "Weight group size W (dimension 1) must be positive.");
        }

        _scales = scales;
        _biases = null;
        _weightGroupSizeH = weightGroupSizeH;
        _weightGroupSizeW = weightGroupSizeW;
    }

    public Tensor Scales => _scales!;

    public Tensor? Biases => _biases;

    public bool IsValid => ScalesGroupCountH > 0 && ScalesGroupCountW > 0;

    public bool HasBiases => _biases != null && IsValid;

    public long WeightGroupSizeH => _weightGroupSizeH;

    public long WeightGroupSizeW => _weightGroupSizeW;

    public long ScalesGroupCountH => _scales!.Dimensions[0];

    public long ScalesGroupCountW => _scales!.Dimensions[1];

    public long BiasesGroupCountH => _biases?.Dimensions[0] ?? 0;

    public long BiasesGroupCountW => _biases?.Dimensions[1] ?? 0;

    public long TotalScaleEntries => (long)ScalesGroupCountH * ScalesGroupCountW;

    public long TotalBiasEntries => HasBiases ? (long)BiasesGroupCountH * BiasesGroupCountW : 0;

    // --- Get Scale/Bias By Group Index ---
    public float GetScaleByGroup(int groupIndexH, int groupIndexW)
    {
        if (groupIndexH < 0 || groupIndexH >= ScalesGroupCountH)
        {
            throw new ArgumentOutOfRangeException(nameof(groupIndexH), $"Group H-index is out of range. Expected 0 to {ScalesGroupCountH - 1}, got {groupIndexH}.");
        }

        if (groupIndexW < 0 || groupIndexW >= ScalesGroupCountW)
        {
            throw new ArgumentOutOfRangeException(nameof(groupIndexW), $"Group W-index is out of range. Expected 0 to {ScalesGroupCountW - 1}, got {groupIndexW}.");
        }

        return (float)_scales![groupIndexH, groupIndexW];
    }

    public float GetBiasByGroup(int groupIndexH, int groupIndexW)
    {
        if (!HasBiases)
        {
            throw new InvalidOperationException("Bias information is not available (e.g., symmetric quantization).");
        }

        // _biases will not be null here due to HasBiases check
        if (groupIndexH < 0 || groupIndexH >= BiasesGroupCountH)
        {
            throw new ArgumentOutOfRangeException(nameof(groupIndexH), $"Group H-index is out of range for biases. Expected 0 to {BiasesGroupCountH - 1}, got {groupIndexH}.");
        }

        if (groupIndexW < 0 || groupIndexW >= BiasesGroupCountW)
        {
            throw new ArgumentOutOfRangeException(nameof(groupIndexW), $"Group W-index is out of range for biases. Expected 0 to {BiasesGroupCountW - 1}, got {groupIndexW}.");
        }

        return (float)_biases![groupIndexH, groupIndexW]; // Use ! to assert _biases is not null after HasBiases check
    }

    // --- Get Scale/Bias By Original Weight Element Index ---
    public float GetScaleByIndex(long originalWeightIndexH, long originalWeightIndexW)
    {
        if (originalWeightIndexH < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalWeightIndexH), "Original weight H-index must be non-negative.");
        }

        if (originalWeightIndexW < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalWeightIndexW), "Original weight W-index must be non-negative.");
        }

        int groupIndexH = (int)(originalWeightIndexH / _weightGroupSizeH);
        int groupIndexW = (int)(originalWeightIndexW / _weightGroupSizeW);
        return GetScaleByGroup(groupIndexH, groupIndexW);
    }

    public float GetBiasByIndex(long originalWeightIndexH, long originalWeightIndexW)
    {
        if (!HasBiases)
        {
            throw new InvalidOperationException("Bias information is not available (e.g., symmetric quantization).");
        }

        if (originalWeightIndexH < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalWeightIndexH), "Original weight H-index must be non-negative.");
        }

        if (originalWeightIndexW < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalWeightIndexW), "Original weight W-index must be non-negative.");
        }

        int groupIndexH = (int)(originalWeightIndexH / _weightGroupSizeH);
        int groupIndexW = (int)(originalWeightIndexW / _weightGroupSizeW);
        return GetBiasByGroup(groupIndexH, groupIndexW);
    }

    public float[] GetScaleColumn(int colIndex)
    {
        if (colIndex < 0 || colIndex >= ScalesGroupCountW)
        {
            throw new ArgumentOutOfRangeException(nameof(colIndex), "Column index is out of bounds for scales matrix.");
        }

        long rows = ScalesGroupCountH;
        float[] column = new float[rows];
        for (int i = 0; i < rows; i++)
        {
            column[i] = (float)_scales![i, colIndex];
        }

        return column;
    }

    public float[] GetScaleRow(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= ScalesGroupCountH)
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of bounds for scales matrix.");
        }

        long cols = ScalesGroupCountW;
        float[] row = new float[cols];
        for (long j = 0; j < cols; j++)
        {
            row[j] = (float)_scales![rowIndex, j];
        }

        return row;
    }

    public float[] GetBiasColumn(int colIndex)
    {
        if (!HasBiases)
        {
            throw new InvalidOperationException("Bias information is not available (e.g., symmetric quantization).");
        }

        // _biases will not be null here
        if (colIndex < 0 || colIndex >= BiasesGroupCountW)
        {
            throw new ArgumentOutOfRangeException(nameof(colIndex), "Column index is out of bounds for biases matrix.");
        }

        long rows = BiasesGroupCountH;
        float[] columnData = new float[rows];
        for (long i = 0; i < rows; i++)
        {
            columnData[i] = (float)_biases![i, colIndex]; // Use ! to assert _biases is not null
        }

        return columnData;
    }

    public float[] GetBiasRow(int rowIndex)
    {
        if (!HasBiases)
        {
            throw new InvalidOperationException("Bias information is not available (e.g., symmetric quantization).");
        }

        // _biases will not be null here
        if (rowIndex < 0 || rowIndex >= BiasesGroupCountH)
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of bounds for biases matrix.");
        }

        int cols = (int)BiasesGroupCountW; // Will be 0 if BiasesGroupCountW is 0, safe.
        float[] rowData = new float[cols];
        for (int j = 0; j < cols; j++)
        {
            rowData[j] = (float)_biases![rowIndex, j]; // Use ! to assert _biases is not null
        }

        return rowData;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("QuantizedWeightsInfo:");
        sb.AppendLine($"  Scale Matrix Group Counts: [{ScalesGroupCountH}, {ScalesGroupCountW}]");
        if (HasBiases)
        {
            sb.AppendLine($"  Bias Matrix Group Counts : [{BiasesGroupCountH}, {BiasesGroupCountW}]");
        }
        else
        {
            sb.AppendLine("  Bias Information         : Not Available (Assumed Symmetric Quantization or Bias is Zero)");
        }

        sb.AppendLine($"  Weight Group Size H      : {WeightGroupSizeH}");
        sb.AppendLine($"  Weight Group Size W      : {WeightGroupSizeW}");
        sb.AppendLine($"  Total Scale Entries      : {TotalScaleEntries}");
        if (HasBiases)
        {
            sb.AppendLine($"  Total Bias Entries       : {TotalBiasEntries}");
        }

        return sb.ToString();
    }
}
