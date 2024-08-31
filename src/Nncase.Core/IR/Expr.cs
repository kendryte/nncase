// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance.Helpers;
using Nncase.Diagnostics;

namespace Nncase.IR;

/// <summary>
/// Expression's metadata.
/// </summary>
public class IRMetadata
{
    /// <summary>
    /// Gets or sets outputs names.
    /// </summary>
    public IReadOnlyList<string>? OutputNames { get; set; }
}

/// <summary>
/// Expression.
/// </summary>
public abstract partial class Expr : IDisposable
{
    private readonly Expr[] _operands;
    private readonly ConcurrentDictionary<Expr, Unit> _users = new(ReferenceEqualityComparer.Instance);
    private IRType? _checkedType;
    private int? _hashCodeCache;
    private bool _disposedValue;

    internal Expr(IEnumerable<Expr> operands)
    {
        ExprScope.Current?.Add(this);
        _operands = operands.ToArray();
        foreach (var operand in _operands)
        {
            operand.AddUser(this);
        }
    }

    internal Expr(Expr[] operands)
    {
        ExprScope.Current?.Add(this);
        _operands = operands;
        foreach (var operand in _operands)
        {
            operand.AddUser(this);
        }
    }

    public IRMetadata Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets checked type.
    /// </summary>
    public IRType CheckedType
    {
        get
        {
            if (_checkedType == null)
            {
                CompilerServices.InferenceType(this);
            }

            Trace.Assert(_checkedType is not null);
            return _checkedType!;
        }

        set
        {
            if (_checkedType != value)
            {
                _checkedType = value;
                InvalidateUsersTypeInference();
            }
        }
    }

    /// <summary>
    /// Gets checked tensor type.
    /// </summary>
    public TensorType CheckedTensorType
    {
        get
        {
            switch (CheckedType)
            {
                case TensorType type:
                    return type;
                case DistributedType type:
                    return type.TensorType;
                default:
                    if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
                    {
                        DumpScope.Current.DumpIR(this, "CheckedTensorType");
                    }

                    throw new InvalidOperationException("Only The Expr Have CheckedType Can Get It's Shape");
            }
        }
    }

    /// <summary>
    /// Gets checked shape.
    /// </summary>
    public Shape CheckedShape
    {
        get
        {
            switch (CheckedType)
            {
                case TensorType type:
                    return type.Shape;
                case DistributedType type:
                    return type.TensorType.Shape;
                default:
                    if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
                    {
                        DumpScope.Current.DumpIR(this, "CheckedShapeError");
                    }

                    throw new InvalidOperationException("Only The Expr Have CheckedType Can Get It's Shape");
            }
        }
    }

    /// <summary>
    /// Gets if this expr is tensortype, can return the checkedDatatype.
    /// </summary>
    public DataType CheckedDataType
    {
        get
        {
            switch (CheckedType)
            {
                case TensorType type:
                    return type.DType;
                case DistributedType type:
                    return type.TensorType.DType;
                default:
                    if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
                    {
                        DumpScope.Current.DumpIR(this, "CheckedDatatypeError");
                    }

                    throw new InvalidOperationException($"{CheckedType} haven't data type");
            }
        }
    }

    /// <summary>
    /// Gets users.
    /// </summary>
    public IEnumerable<Expr> Users => EnsureAlive()._users.Keys;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public ReadOnlySpan<Expr> Operands => EnsureAlive()._operands;

    /// <summary>
    /// Gets a value indicating whether the expr is alive.
    /// </summary>
    public bool IsAlive => !_disposedValue;

    /// <summary>
    /// Gets or sets raw checked type.
    /// </summary>
    internal IRType? RawCheckedType
    {
        get => _checkedType;
        set => _checkedType = value;
    }

    public static bool operator ==(Expr? left, Expr? right) => EqualityComparer<Expr>.Default.Equals(left, right);

    public static bool operator !=(Expr? left, Expr? right) => !(left == right);

    /// <summary>
    /// Accept a <see cref="ExprFunctor{TExprResult, TTypeResult, TContext}"/>.
    /// </summary>
    /// <typeparam name="TExprResult">Result type of visiting expressions.</typeparam>
    /// <typeparam name="TTypeResult">Result type of visiting types.</typeparam>
    /// <typeparam name="TContext">Visit context.</typeparam>
    /// <param name="functor">Expression functor.</param>
    /// <param name="context">Context.</param>
    /// <returns>Visit result.</returns>
    public abstract TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context);

    /// <inheritdoc/>
    public override string ToString()
    {
        return GetType().ToString();
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is Expr other
            && GetType() == other.GetType()
            && GetHashCode() == other.GetHashCode()
            && Operands.SequenceEqual(other.Operands);
    }

    /// <inheritdoc/>
    public sealed override int GetHashCode() => _hashCodeCache ??= GetHashCodeCore();

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    public void DisposeIfNoUsers()
    {
        if (_users.Keys.Count == 0)
        {
            Dispose();
        }
    }

    internal void AddUser(Expr user)
    {
        EnsureAlive();
        Trace.Assert(!ReferenceEquals(this, user));
        _users.TryAdd(user.EnsureAlive(), default);
    }

    internal void RemoveUser(Expr user)
    {
        _users.Remove(user, out _);
    }

    internal void ReplaceOperand(int index, Expr newOperand)
    {
        ref var operand = ref _operands[index];
        if (!ReferenceEquals(operand, newOperand))
        {
            newOperand.AddUser(this);
            operand.RemoveUser(this);
            operand = newOperand;
            OnOperandsReplaced();
        }
    }

    internal void ReplaceAllUsesWith(Expr newOperand)
        => ReplaceScopedUsesWith(newOperand, null);

    internal void ReplaceScopedUsesWith(Expr newOperand, IReadOnlySet<Expr>? scope)
    {
        EnsureAlive();
        if (!ReferenceEquals(this, newOperand))
        {
            foreach (var user in Users.ToArray())
            {
                if ((scope is null || scope.Contains(user))
                    && !newOperand.IsDescendantOf(this))
                {
                    newOperand.AddUser(user);
                    var operands = user._operands;
                    for (int i = 0; i < operands.Length; i++)
                    {
                        ref var operand = ref operands[i];
                        if (ReferenceEquals(operand, this))
                        {
                            operand = newOperand;
                        }
                    }

                    user.OnOperandsReplaced();
                    RemoveUser(user);
                }
            }
        }
    }

    protected virtual int GetHashCodeCore()
    {
        return HashCode.Combine(GetType(), HashCode<Expr>.Combine(Operands));
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            foreach (var operand in _operands)
            {
                operand.RemoveUser(this);
                operand.DisposeIfNoUsers();
            }

            _disposedValue = true;
        }
    }

    private bool IsDescendantOf(Expr other, Dictionary<Expr, bool> visited)
    {
        if (visited.TryGetValue(this, out var result))
        {
            return result;
        }

        foreach (var operand in _operands)
        {
            if (ReferenceEquals(operand, other))
            {
                result = true;
                break;
            }
        }

        foreach (var operand in _operands)
        {
            if (operand.IsDescendantOf(other, visited))
            {
                result = true;
                break;
            }
        }

        visited.Add(this, result);
        return result;
    }

    private bool IsDescendantOf(Expr other)
    {
        return IsDescendantOf(other, new Dictionary<Expr, bool>(ReferenceEqualityComparer.Instance));
    }

    private void OnOperandsReplaced()
    {
        InvalidateTypeInference();
        InvalidateHashCodeCache();
    }

    private void InvalidateTypeInference()
    {
        if (_checkedType != null)
        {
            _checkedType = null;
            InvalidateUsersTypeInference();
        }
    }

    private void InvalidateUsersTypeInference()
    {
        foreach (var user in Users)
        {
            user.InvalidateTypeInference();
        }
    }

    private void InvalidateHashCodeCache()
    {
        if (_hashCodeCache != null)
        {
            _hashCodeCache = null;
            InvalidateUsersHashCodeCache();
        }
    }

    private void InvalidateUsersHashCodeCache()
    {
        foreach (var user in Users)
        {
            user.InvalidateHashCodeCache();
        }
    }

    private Expr EnsureAlive()
    {
        if (_disposedValue)
        {
            throw new ObjectDisposedException(null);
        }

        return this;
    }
}
