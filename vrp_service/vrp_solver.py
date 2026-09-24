from flask import Flask, request, jsonify
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

app = Flask(__name__)

@app.route('/solve_vrp', methods=['POST'])
def solve_vrp():
    """
    Expects JSON payload:
    {
      "distance_matrix": [[0, 10, 20], [10, 0, 15], [20, 15, 0]],
      "num_vehicles": 2,
      "depot": 0
    }
    """
    data = request.json
    distance_matrix = data.get('distance_matrix')
    num_vehicles = data.get('num_vehicles', 1)
    depot = data.get('depot', 0)

    if not distance_matrix:
        return jsonify({"error": "distance_matrix is required"}), 400

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix), num_vehicles, depot
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint to balance workload across vehicles.
    # We allow a very large max distance to ensure a solution is found,
    # but we add a span cost coefficient to balance the routes.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000000,  # maximum travel distance/time
        True,  # start cumul to zero
        dimension_name)
    
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100) # Highly encourages balanced routes

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(2) # Give it 2 seconds to optimize

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return jsonify({"error": "No solution found by OR-Tools"}), 400

    # Extract routes
    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        # Add the final return to depot node if desired, but we'll exclude the end depot for the raw output
        routes.append(route)

    return jsonify({
        "status": "success",
        "routes": routes
    })

if __name__ == '__main__':
    print("Starting OR-Tools VRP Solver on port 8003...")
    app.run(host='0.0.0.0', port=8003)
