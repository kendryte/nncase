// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Google.OrTools.ConstraintSolver;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;

namespace Nncase.Tests.EGraphTest;

internal interface IDataModel
{
    long[,] DistanceMatrix { get; }

    int VehicleNumber { get; }

    int Depot { get; }
}

[AutoSetupTestMethod(InitSession = false)]
public class UnitTestVrp : TestClassBase
{
    /// <summary>
    ///               x
    ///            /     \
    ///         /         \
    ///      /            conv  --------
    ///    |           /   |   \        |
    ///    |         /     |    \       |
    /// convact    clamp  relu6  act    |
    ///              |                  |
    ///             conv2d              |
    ///              |                  |
    ///             add ----------------.
    /// </summary>
    [Fact]
    public void TestSimpleVrp()
    {
        // Instantiate the data problem.
        var data = new DataModel();

        // Create Routing Index Manager
        var manager = new RoutingIndexManager(data.DistanceMatrix.GetLength(0), data.VehicleNumber, data.Depot);

        // Create Routing Model.
        var routing = new RoutingModel(manager);

        // Create and register a transit callback.
        int transitCallbackIndex = routing.RegisterTransitCallback((long fromIndex, long toIndex) =>
                                                                   {
                                                                       // Convert from routing variable Index to
                                                                       // distance matrix NodeIndex.
                                                                       var fromNode = manager.IndexToNode(fromIndex);
                                                                       var toNode = manager.IndexToNode(toIndex);
                                                                       return data.DistanceMatrix[fromNode, toNode];
                                                                   });

        // Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transitCallbackIndex);

        // Add Distance constraint.
        routing.AddDimension(transitCallbackIndex, 0, 3000, true, "Distance");
        RoutingDimension distanceDimension = routing.GetMutableDimension("Distance");
        distanceDimension.SetGlobalSpanCostCoefficient(100);

        // Setting first solution heuristic.
        RoutingSearchParameters searchParameters =
            operations_research_constraint_solver.DefaultRoutingSearchParameters();
        searchParameters.FirstSolutionStrategy = FirstSolutionStrategy.Types.Value.PathCheapestArc;

        // Solve the problem.
        Assignment solution = routing.SolveWithParameters(searchParameters);

        // Print solution on console.
        PrintSolution(data, routing, manager, solution);
    }

    [Fact]
    public void TestSimpleEGraphVrp()
    {
        var datamodel = new DataModel2();
        var manager = new RoutingIndexManager(datamodel.DistanceMatrix.GetLength(0), datamodel.VehicleNumber, new[] { 0 }, new[] { 4 });

        // Create Routing Model.
        var routing = new RoutingModel(manager);

        // Create and register a transit callback.
        int transitCallbackIndex = routing.RegisterTransitCallback((long fromIndex, long toIndex) =>
                                                                   {
                                                                       // Convert from routing variable Index to
                                                                       // distance matrix NodeIndex.
                                                                       var fromNode = manager.IndexToNode(fromIndex);
                                                                       var toNode = manager.IndexToNode(toIndex);
                                                                       return datamodel.DistanceMatrix[fromNode, toNode];
                                                                   });

        // Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transitCallbackIndex);

        RoutingSearchParameters searchParameters =
            operations_research_constraint_solver.DefaultRoutingSearchParameters();
        searchParameters.FirstSolutionStrategy = FirstSolutionStrategy.Types.Value.PathCheapestArc;

        // Solve the problem.
        Assignment solution = routing.SolveWithParameters(searchParameters);

        // Print solution on console.
        PrintSolution(datamodel, routing, manager, solution);

        // result : 0 -> 1 -> 2 -> 3 -> 4.
    }

    [Fact]
    public void TestSimpleEgraphSat()
    {
        var model = new CpModel();
        var vars = new[] { model.NewBoolVar("x"),
          model.NewBoolVar("conv2d_1"),
          model.NewBoolVar("conv2dAct"),
          model.NewBoolVar("clamp"),
          model.NewBoolVar("relu6"),
          model.NewBoolVar("act"),
          model.NewBoolVar("conv2d_2"),
          model.NewBoolVar("add"),
        };

        // 1. must pick one from the root.
        model.AddExactlyOne(new[] { vars[^1] });

        // 2. must pick one from the chilren eclass nodes.
        model.AddBoolOr(new[] { vars[^1].Not(), vars[1] }); // add 有两个children
        model.AddBoolOr(new[] { vars[^1].Not(), vars[^2] });
        model.AddBoolOr(new[] { vars[^2].Not(), vars[^3], vars[^4], vars[^5], vars[^6] });
        model.AddBoolOr(new[] { vars[^3].Not(), vars[1] });
        model.AddBoolOr(new[] { vars[^4].Not(), vars[1] });
        model.AddBoolOr(new[] { vars[^5].Not(), vars[1] });
        model.AddBoolOr(new[] { vars[^6].Not(), vars[0] });
        model.AddBoolOr(new[] { vars[1].Not(), vars[0] });

        // 3. pick less nodes
        var costs = new[] { 0, 100, 125, 50, 30, 40, 100, 20 };
        var obj = LinearExpr.WeightedSum(vars, costs);
        model.Minimize(obj);

        // 3. soft clause
        var solver = new CpSolver();
        solver.StringParameters = "enumerate_all_solutions:true";
#if DEBUG
        System.Console.WriteLine(model.Validate());
#endif
        var status = solver.Solve(model, new PrintCallBack(vars, costs));
        if (status is CpSolverStatus.Feasible or CpSolverStatus.Optimal)
        {
            foreach (var v in vars)
            {
#if DEBUG
                System.Console.WriteLine(v.Name() + " " + solver.BooleanValue(v));
#endif
            }

            Assert.True(solver.BooleanValue(vars[0]));
            Assert.True(solver.BooleanValue(vars[1]));
            Assert.False(solver.BooleanValue(vars[2]));
            Assert.False(solver.BooleanValue(vars[3]));
            Assert.True(solver.BooleanValue(vars[4]));
            Assert.False(solver.BooleanValue(vars[5]));
            Assert.True(solver.BooleanValue(vars[6]));
            Assert.True(solver.BooleanValue(vars[7]));
        }
    }

    [Fact]
    public void TestOverLap()
    {
        // note ortools no overlap not support 0 size.
        var model = new CpModel();

        var x0 = model.NewIntervalVar(model.NewConstant(0), model.NewConstant(2), model.NewConstant(2), "x0");
        var y0 = model.NewFixedSizeIntervalVar(model.NewIntVar(0, 10, "y0_start"), 7, "y0");

        var x1 = model.NewIntervalVar(model.NewConstant(2), model.NewConstant(0), model.NewConstant(2), "x1");
        var y1 = model.NewFixedSizeIntervalVar(model.NewIntVar(0, 10, "y1_start"), 7, "y1");

        var x2 = model.NewIntervalVar(model.NewConstant(2), model.NewConstant(1), model.NewConstant(3), "x2");
        var y2 = model.NewFixedSizeIntervalVar(model.NewIntVar(0, 10, "y2_start"), 7, "y2");

        model.Add(y0.StartExpr() == y1.StartExpr());
        model.Add(y1.StartExpr() == y2.StartExpr());
        var nooverlap = model.AddNoOverlap2D();
        nooverlap.AddRectangle(x0, y0);
        nooverlap.AddRectangle(x1, y1);
        nooverlap.AddRectangle(x2, y2);
        model.Minimize(y0.StartExpr() + y1.StartExpr() + y2.StartExpr());

        var solver = new CpSolver();
        var status = solver.Solve(model);

        Assert.Equal(CpSolverStatus.Infeasible, status);
    }

    private static void PrintSolution(in IDataModel data, in RoutingModel routing, in RoutingIndexManager manager, in Assignment solution)
    {
#if DEBUG
        Console.WriteLine($"Objective {solution.ObjectiveValue()}:");

        // Inspect solution.
        long maxRouteDistance = 0;
        for (int i = 0; i < data.VehicleNumber; ++i)
        {
            Console.WriteLine("Route for Vehicle {0}:", i);
            long routeDistance = 0;
            var index = routing.Start(i);
            while (routing.IsEnd(index) == false)
            {
                Console.Write("{0} -> ", manager.IndexToNode((int)index));
                var previousIndex = index;
                index = solution.Value(routing.NextVar(index));
                routeDistance += routing.GetArcCostForVehicle(previousIndex, index, 0);
            }

            Console.WriteLine("{0}", manager.IndexToNode((int)index));
            Console.WriteLine("Distance of the route: {0}m", routeDistance);
            maxRouteDistance = Math.Max(routeDistance, maxRouteDistance);
        }

        Console.WriteLine("Maximum distance of the routes: {0}m", maxRouteDistance);
#endif
    }

    private class PrintCallBack : CpSolverSolutionCallback
    {
        private readonly BoolVar[] _vars;
        private readonly int[] _costs;
        private int _count;

        public PrintCallBack(BoolVar[] vars, int[] costs)
        {
            _vars = vars;
            _costs = costs;
            _count = 0;
        }

        public override void OnSolutionCallback()
        {
#if DEBUG
            System.Console.WriteLine($"Solution {_count++}");
            foreach (var v in _vars)
            {
                System.Console.WriteLine(v.Name() + " " + BooleanValue(v));
            }

            System.Console.WriteLine("costs: " + _vars.Zip(_costs).Select(p => BooleanValue(p.First) ? p.Second : 0).Sum());
            System.Console.WriteLine();
#else
            _count++;
#endif
        }
    }
}

internal class DataModel : IDataModel
{
    public long[,] DistanceMatrix { get; } = {
            { 0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662 },
            { 548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210 },
            { 776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754 },
            { 696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358 },
            { 582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244 },
            { 274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708 },
            { 502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480 },
            { 194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856 },
            { 308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514 },
            { 194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468 },
            { 536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354 },
            { 502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844 },
            { 388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730 },
            { 354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536 },
            { 468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194 },
            { 776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798 },
            { 662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0 },
        };

    public int VehicleNumber { get; } = 4;

    public int Depot { get; }
}

internal class DataModel2 : IDataModel
{
    public long[,] DistanceMatrix { get; } = {
            { 0, 100, 125, long.MaxValue, long.MaxValue, },
            { 100, 0, 50, long.MaxValue, long.MaxValue, },
            { 125, 50, 0, 50, long.MaxValue, },
            { long.MaxValue, long.MaxValue, 50, 0, 30, },
            { long.MaxValue, long.MaxValue, long.MaxValue, 30, 0, },
        };

    public int VehicleNumber { get; } = 1;

    public int Depot { get; }
}
