"""Bounded v5 Mixed Workflow solver. Matrices order vehicle depots, then tasks."""

from __future__ import annotations

import math
import time

from flask import Flask, jsonify, request
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

app = Flask(__name__)
SECOND = 60
LITRE = 1000
CENTIKG = 100


class ContractError(ValueError):
    pass


def number(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < minimum:
        raise ContractError(f"{field} must be a finite number >= {minimum}")
    return value


def scaled(value, scale):
    return round(value * scale)


def point(value, field):
    if not isinstance(value, dict):
        raise ContractError(f"{field} requires lat/lng")
    lat = number(value.get("lat"), f"{field}.lat", -90)
    lng = number(value.get("lng"), f"{field}.lng", -180)
    if lat > 90 or lng > 180:
        raise ContractError(f"{field} coordinates are out of range")


def validate(data):
    if not isinstance(data, dict) or data.get("contract_version") != "v5":
        raise ContractError("contract_version must be v5")
    vehicles, tasks, options = data.get("vehicles"), data.get("tasks"), data.get("options", {})
    if not isinstance(vehicles, list) or not vehicles or not isinstance(tasks, list) or not isinstance(options, dict):
        raise ContractError("vehicles, tasks and options have invalid shapes")
    limit = number(options.get("solver_time_limit_seconds", options.get("time_limit_s", 20)), "solver_time_limit_seconds", 2)
    if limit > 60 or len(tasks) > number(options.get("max_candidate_nodes", 200), "max_candidate_nodes", 1):
        raise ContractError("solver time or candidate count exceeds configured bound")
    size = len(vehicles) + len(tasks)
    matrices = []
    for name in ("travel_time_matrix_min", "distance_matrix_m"):
        matrix = data.get(name)
        if not isinstance(matrix, list) or len(matrix) != size or any(not isinstance(row, list) or len(row) != size for row in matrix):
            raise ContractError(f"{name} must be a {size}x{size} matrix")
        for row in matrix:
            for cell in row:
                number(cell, name)
        matrices.append(matrix)
    for entries, name in ((vehicles, "vehicles"), (tasks, "tasks")):
        ids = [entry.get("id") for entry in entries]
        if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            raise ContractError(f"{name} need unique nonempty ids")
    for vehicle in vehicles:
        point(vehicle.get("depot"), "vehicle.depot")
        for key in ("shift_start_min", "shift_end_min", "break_min", "max_tasks", "speed_factor", "service_time_factor"):
            number(vehicle.get(key), key, 0.01 if key.endswith("factor") else 0)
        if int(vehicle["max_tasks"]) != vehicle["max_tasks"]:
            raise ContractError("max_tasks must be an integer")
        if vehicle["shift_start_min"] + vehicle["break_min"] >= vehicle["shift_end_min"]:
            raise ContractError("vehicle shift must exceed break")
        if not isinstance(vehicle.get("allowed_task_types"), list):
            raise ContractError("allowed_task_types must be an array")
        if not isinstance(vehicle.get("certifications", []), list) or any(not isinstance(c, str) for c in vehicle.get("certifications", [])):
            raise ContractError("vehicle.certifications must be strings")
    task_ids = {task["id"] for task in tasks}
    for task in tasks:
        point(task.get("location"), "task.location")
        if not isinstance(task.get("task_type"), str):
            raise ContractError("task_type is required")
        number(task.get("service_minutes"), "service_minutes")
        number(task.get("drop_penalty"), "drop_penalty", 1)
        if not isinstance(task.get("required_certifications", []), list) or any(not isinstance(c, str) for c in task.get("required_certifications", [])):
            raise ContractError("required_certifications must be strings")
        if task.get("anchor_id") and task["anchor_id"] not in task_ids:
            raise ContractError("anchor_id must name a task")
        if task.get("anchor_id") == task["id"] or (task.get("anchor_id") and next(t for t in tasks if t["id"] == task["anchor_id"]).get("anchor_id")):
            raise ContractError("satellite anchor_id must name a root anchor")
        first = number(task.get("time_window_start_min", 0), "time_window_start_min")
        last = number(task.get("time_window_end_min", max(v["shift_end_min"] for v in vehicles)), "time_window_end_min")
        if first > last:
            raise ContractError("task time window is reversed")
    return vehicles, tasks, options, matrices[0], matrices[1], limit


def eligible_vehicles(task, vehicles, travel, node):
    if any(task.get(key) is None for key in ("estimated_volume_m3", "estimated_weight_kg")):
        return [], "unknown_load"
    for key in ("estimated_volume_m3", "estimated_weight_kg"):
        number(task[key], key)
    eligible, defects = [], []
    for index, vehicle in enumerate(vehicles):
        if task["task_type"] not in vehicle["allowed_task_types"]:
            defects.append("no_capable_vehicle")
            continue
        held = set(vehicle.get("certifications", []))
        if vehicle.get("hazmat_certified"):
            held.add("hazmat")
        missing = set(task.get("required_certifications", [])) - held
        if missing:
            defects.append("hazmat_certification" if "hazmat" in missing else "no_capable_vehicle")
            continue
        load_keys = ("max_volume_m3", "max_weight_kg", "starting_volume_m3", "starting_weight_kg")
        if any(vehicle.get(key) is None for key in load_keys):
            defects.append("unknown_load")
            continue
        for key in load_keys:
            number(vehicle[key], key)
        if vehicle["starting_volume_m3"] >= vehicle["max_volume_m3"] and task["estimated_volume_m3"] > 0:
            defects.append("capacity_volume")
        elif vehicle["starting_weight_kg"] >= vehicle["max_weight_kg"] and task["estimated_weight_kg"] > 0:
            defects.append("capacity_weight")
        elif vehicle["starting_volume_m3"] + task["estimated_volume_m3"] > vehicle["max_volume_m3"]:
            defects.append("capacity_volume")
        elif vehicle["starting_weight_kg"] + task["estimated_weight_kg"] > vehicle["max_weight_kg"]:
            defects.append("capacity_weight")
        elif vehicle["max_tasks"] < (0 if task.get("anchor_id") else 1):
            defects.append("max_tasks")
        else:
            work = task["service_minutes"] * vehicle["service_time_factor"]
            trip = (travel[index][node] + travel[node][index]) / vehicle["speed_factor"]
            earliest = max(vehicle["shift_start_min"] + travel[index][node] / vehicle["speed_factor"], task.get("time_window_start_min", 0))
            latest = min(vehicle["shift_end_min"] - vehicle["break_min"], task.get("time_window_end_min", math.inf))
            if earliest > latest or trip + work > vehicle["shift_end_min"] - vehicle["shift_start_min"] - vehicle["break_min"]:
                defects.append("shift_exceeded")
            else:
                eligible.append(index)
    for reason in ("unknown_load", "hazmat_certification", "capacity_volume", "capacity_weight", "shift_exceeded", "max_tasks"):
        if reason in defects and not eligible:
            return [], reason
    return eligible, None if eligible else "no_capable_vehicle"


def solve_v5(data):
    vehicles, tasks, options, travel, distance, limit = validate(data)
    started = time.monotonic()
    count = len(vehicles)
    nodes = {task["id"]: count + i for i, task in enumerate(tasks)}
    manager = pywrapcp.RoutingIndexManager(count + len(tasks), count, list(range(count)), list(range(count)))
    routing = pywrapcp.RoutingModel(manager)
    solver = routing.solver()
    callbacks = []
    preference = number(options.get("generalist_sweeper_penalty", 0.25), "generalist_sweeper_penalty")
    for vehicle_index, vehicle in enumerate(vehicles):
        def transit(source, target, vehicle=vehicle):
            a, b = manager.IndexToNode(source), manager.IndexToNode(target)
            service = tasks[a - count]["service_minutes"] if a >= count else 0
            travel_seconds = travel[a][b] * SECOND / vehicle["speed_factor"]
            # Preference changes cost only; a separate callback constrains real time.
            return math.ceil(travel_seconds + service * SECOND * vehicle["service_time_factor"])
        callback = routing.RegisterTransitCallback(transit)
        callbacks.append(callback)
        def cost(source, target, transit=transit, vehicle=vehicle):
            b = manager.IndexToNode(target)
            extra = preference if b >= count and tasks[b - count]["task_type"] == "Sweeper" and vehicle.get("supports_standard") else 0
            return math.ceil(transit(source, target) * (1 + extra))
        routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(cost), vehicle_index)
    horizon = math.ceil(max(v["shift_end_min"] for v in vehicles) * SECOND)
    routing.AddDimensionWithVehicleTransits(callbacks, horizon, horizon, False, "Time")
    clock = routing.GetDimensionOrDie("Time")
    for i, vehicle in enumerate(vehicles):
        clock.CumulVar(routing.Start(i)).SetValue(scaled(vehicle["shift_start_min"], SECOND))
        clock.CumulVar(routing.End(i)).SetMax(scaled(vehicle["shift_end_min"] - vehicle["break_min"], SECOND))
    for task in tasks:
        index = manager.NodeToIndex(nodes[task["id"]])
        first = min(horizon, scaled(task.get("time_window_start_min", 0), SECOND))
        last = min(horizon, scaled(task.get("time_window_end_min", horizon / SECOND), SECOND))
        clock.CumulVar(index).SetRange(first, last)

    def load_dimension(name, demand_key, capacity_key, starting_key, scale):
        def demand(index):
            node = manager.IndexToNode(index)
            return scaled(tasks[node - count].get(demand_key) or 0, scale) if node >= count else 0
        callback = routing.RegisterUnaryTransitCallback(demand)
        routing.AddDimensionWithVehicleCapacity(callback, 0,
                                                [scaled(max(vehicle.get(capacity_key) or 0, vehicle.get(starting_key) or 0), scale) for vehicle in vehicles],
                                                False, name)
        dimension = routing.GetDimensionOrDie(name)
        for i, vehicle in enumerate(vehicles):
            dimension.CumulVar(routing.Start(i)).SetValue(scaled(vehicle.get(starting_key) or 0, scale))
        return dimension
    volume = load_dimension("Volume", "estimated_volume_m3", "max_volume_m3", "starting_volume_m3", LITRE)
    weight = load_dimension("Weight", "estimated_weight_kg", "max_weight_kg", "starting_weight_kg", CENTIKG)
    count_callback = routing.RegisterUnaryTransitCallback(
        lambda index: int(manager.IndexToNode(index) >= count and
                          not tasks[manager.IndexToNode(index) - count].get("anchor_id")))
    routing.AddDimensionWithVehicleCapacity(count_callback, 0, [int(v["max_tasks"]) for v in vehicles], True, "TaskCount")
    order_callback = routing.RegisterTransitCallback(lambda _source, _target: 1)
    routing.AddDimension(order_callback, 0, count + len(tasks) + 1, True, "Order")
    order = routing.GetDimensionOrDie("Order")

    preexcluded = {}
    for task in tasks:
        index = manager.NodeToIndex(nodes[task["id"]])
        allowed, reason = eligible_vehicles(task, vehicles, travel, nodes[task["id"]])
        routing.AddDisjunction([index], round(task["drop_penalty"]))
        if allowed:
            # Restrict the routing variable directly. OR-Tools 9.15's Python 3.14
            # binding rejects the Span argument of SetAllowedVehiclesForIndex.
            routing.VehicleVar(index).SetValues([-1, *allowed])
        else:
            routing.ActiveVar(index).SetValue(0)
            preexcluded[task["id"]] = reason

    bundles = {}
    for task in tasks:
        anchor_id = task.get("anchor_id")
        if not anchor_id:
            continue
        bundles.setdefault(task.get("bundle_group_key") or anchor_id, []).append(task["id"])
        anchor = manager.NodeToIndex(nodes[anchor_id])
        satellite = manager.NodeToIndex(nodes[task["id"]])
        active = routing.ActiveVar(satellite)
        solver.Add(active <= routing.ActiveVar(anchor))
        solver.Add(routing.VehicleVar(satellite) - routing.VehicleVar(anchor) <= count * (1 - active))
        solver.Add(routing.VehicleVar(anchor) - routing.VehicleVar(satellite) <= count * (1 - active))
        solver.Add(order.CumulVar(anchor) + 1 <= order.CumulVar(satellite) + (count + len(tasks) + 1) * (1 - active))
    for members in bundles.values():
        first = routing.ActiveVar(manager.NodeToIndex(nodes[members[0]]))
        for member in members[1:]:
            solver.Add(first == routing.ActiveVar(manager.NodeToIndex(nodes[member])))

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromMilliseconds(round(limit * 1000))
    solution = routing.SolveWithParameters(parameters)
    elapsed = round(time.monotonic() - started, 3)
    if solution is None:
        return {"contract_version": "v5", "status": "failed", "routes": [],
                "unassigned": [{"task_id": task["id"], "reason": preexcluded.get(task["id"], "dropped_by_solver")} for task in tasks],
                "solver_diagnostics": {"error_code": "solver_no_solution", "elapsed_seconds": elapsed}}

    routes, assigned = [], set()
    for i, vehicle in enumerate(vehicles):
        index, previous, stops, metres = routing.Start(i), i, [], 0
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(next_index)
            leg = round(distance[previous][node])
            metres += leg
            if node >= count:
                task = tasks[node - count]
                arrival = solution.Value(clock.CumulVar(next_index)) / SECOND
                stops.append({"task_id": task["id"], "arrival_min": round(arrival, 2),
                              "depart_min": round(arrival + task["service_minutes"] * vehicle["service_time_factor"], 2),
                              "distance_from_prev_m": leg,
                              "time_from_prev_min": round(travel[previous][node] / vehicle["speed_factor"], 2),
                              "volume_after_m3": round((solution.Value(volume.CumulVar(next_index)) + scaled(task["estimated_volume_m3"], LITRE)) / LITRE, 3),
                              "weight_after_kg": round((solution.Value(weight.CumulVar(next_index)) + scaled(task["estimated_weight_kg"], CENTIKG)) / CENTIKG, 2)})
                assigned.add(task["id"])
            previous, index = node, next_index
        routes.append({"vehicle_id": vehicle["id"], "stops": stops, "total_distance_m": metres,
                       "total_duration_min": round((solution.Value(clock.CumulVar(index)) - solution.Value(clock.CumulVar(routing.Start(i)))) / SECOND, 2)})
    unassigned = []
    for task in tasks:
        if task["id"] in assigned:
            continue
        reason = preexcluded.get(task["id"])
        if reason is None:
            possible, _ = eligible_vehicles(task, vehicles, travel, nodes[task["id"]])
            if possible and all((routes[i]["stops"][-1]["volume_after_m3"] if routes[i]["stops"] else vehicles[i]["starting_volume_m3"])
                                + task["estimated_volume_m3"] > vehicles[i]["max_volume_m3"] for i in possible):
                reason = "capacity_volume"
            elif possible and all((routes[i]["stops"][-1]["weight_after_kg"] if routes[i]["stops"] else vehicles[i]["starting_weight_kg"])
                                  + task["estimated_weight_kg"] > vehicles[i]["max_weight_kg"] for i in possible):
                reason = "capacity_weight"
            else:
                reason = "dropped_by_solver"
        unassigned.append({"task_id": task["id"], "reason": reason})
    return {"contract_version": "v5", "status": "success", "routes": routes, "unassigned": unassigned,
            "solver_diagnostics": {"elapsed_seconds": elapsed, "candidate_nodes": len(tasks),
                                   "time_limit_seconds": limit, "dropped_count": len(unassigned)}}


@app.route("/solve_vrp", methods=["POST"])
def solve_vrp():
    try:
        return jsonify(solve_v5(request.get_json(silent=True)))
    except ContractError as error:
        return jsonify({"contract_version": "v5", "error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003)
