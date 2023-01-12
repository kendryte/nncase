// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc;

namespace Nncase.Hosting;

/// <summary>
/// Application part.
/// </summary>
public interface IApplicationPart
{
    /// <summary>
    /// Configure services.
    /// </summary>
    /// <param name="registrator">Service registrator.</param>
    void ConfigureServices(IRegistrator registrator);
}

/// <summary>
/// Application part extensions.
/// </summary>
public static class ApplicationPartExtensions
{
    /// <summary>
    /// Register module.
    /// </summary>
    /// <typeparam name="TModule">Module type.</typeparam>
    /// <param name="registrator">Service registrator.</param>
    /// <returns>Configured service registrator.</returns>
    public static IRegistrator RegisterModule<TModule>(this IRegistrator registrator)
        where TModule : class, IApplicationPart, new()
    {
        var module = new TModule();
        module.ConfigureServices(registrator);
        return registrator;
    }

    /// <summary>Registers single registration for all implemented public interfaces and base classes.</summary>
    /// <typeparam name="TImplementation">Implementation type.</typeparam>
    /// <param name="registrator">Service registrator.</param>
    /// <param name="reuse">Reuse strategy.</param>
    public static void RegisterManyInterface<TImplementation>(this IRegistrator registrator, IReuse? reuse = null)
        where TImplementation : class
    {
        registrator.RegisterMany<TImplementation>(reuse, serviceTypeCondition: t => t.IsInterface);
    }
}
