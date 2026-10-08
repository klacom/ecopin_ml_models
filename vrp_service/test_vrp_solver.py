import copy
import unittest

from vrp_solver import app, solve_v5


def vehicle(id="truck", **changes):
    return {"id": id, "depot": {"lat": 14.56, "lng": 121.07},
            "shift_start_min": 480, "shift_end_min": 660, "break_min": 20,
            "max_tasks": 4, "speed_factor": 1, "service_time_factor": 1,
            "allowed_task_types": ["Cleanup", "Sweeper", "Mixed"],
            "hazmat_certified": False, "max_volume_m3": 3, "max_weight_kg": 500,
            "starting_volume_m3": 1, "starting_weight_kg": 50, **changes}


def task(id, **changes):
    return {"id": id, "location": {"lat": 14.57, "lng": 121.08},
            "task_type": "Cleanup", "service_minutes": 10,
            "estimated_volume_m3": 0.5, "estimated_weight_kg": 40,
            "required_certifications": [], "drop_penalty": 100000, **changes}


def payload(vehicles, tasks):
    size = len(vehicles) + len(tasks)
    return {"contract_version": "v5", "vehicles": vehicles, "tasks": tasks,
            "travel_time_matrix_min": [[0 if i == j else 5 for j in range(size)] for i in range(size)],
            "distance_matrix_m": [[0 if i == j else 1000 for j in range(size)] for i in range(size)],
            "options": {"solver_time_limit_seconds": 2, "max_candidate_nodes": 20}}


class SolverContractTest(unittest.TestCase):
    def test_load_and_window_are_enforced(self):
        data = payload([vehicle()], [task("small", time_window_start_min=500, time_window_end_min=550),
                                     task("too_large", estimated_volume_m3=2.5),
                                     task("too_late", time_window_start_min=700, time_window_end_min=720)])
        result = solve_v5(data)
        self.assertEqual(result["contract_version"], "v5")
        self.assertEqual([stop["task_id"] for stop in result["routes"][0]["stops"]], ["small"])
        self.assertGreaterEqual(result["routes"][0]["stops"][0]["arrival_min"], 500)
        self.assertEqual({entry["task_id"]: entry["reason"] for entry in result["unassigned"]},
                         {"too_large": "capacity_volume", "too_late": "shift_exceeded"})

    def test_satellites_follow_anchor_on_same_vehicle_or_all_drop(self):
        data = payload([vehicle("one"), vehicle("two")],
                       [task("anchor", task_type="Mixed"),
                        task("sat1", task_type="Mixed", anchor_id="anchor", bundle_group_key="bundle"),
                        task("sat2", task_type="Mixed", anchor_id="anchor", bundle_group_key="bundle")])
        result = solve_v5(data)
        routes = [[stop["task_id"] for stop in route["stops"]] for route in result["routes"]]
        containing = [route for route in routes if "anchor" in route]
        self.assertEqual(len(containing), 1)
        self.assertLess(containing[0].index("anchor"), containing[0].index("sat1"))
        self.assertLess(containing[0].index("anchor"), containing[0].index("sat2"))
        self.assertEqual(sum("sat1" in route for route in routes), 1)
        self.assertEqual(sum("sat2" in route for route in routes), 1)

    def test_infeasible_satellite_group_drops_without_losing_anchor(self):
        data = payload([vehicle(max_volume_m3=2, starting_volume_m3=1)],
                       [task("anchor", task_type="Mixed", estimated_volume_m3=0.1),
                        task("sat1", task_type="Mixed", anchor_id="anchor", bundle_group_key="bundle", estimated_volume_m3=0.1),
                        task("sat2", task_type="Mixed", anchor_id="anchor", bundle_group_key="bundle", estimated_volume_m3=2)])
        result = solve_v5(data)
        self.assertEqual([stop["task_id"] for stop in result["routes"][0]["stops"]], ["anchor"])
        self.assertEqual({item["task_id"] for item in result["unassigned"]}, {"sat1", "sat2"})

    def test_certification_and_unknown_load_fail_closed(self):
        data = payload([vehicle()], [task("hazmat", required_certifications=["hazmat"]),
                                     task("unknown", estimated_weight_kg=None)])
        result = solve_v5(data)
        self.assertEqual({entry["task_id"]: entry["reason"] for entry in result["unassigned"]},
                         {"hazmat": "hazmat_certification", "unknown": "unknown_load"})

    def test_cumulative_volume_and_starting_load(self):
        data = payload([vehicle(max_volume_m3=2, starting_volume_m3=1)],
                       [task("first", estimated_volume_m3=0.75),
                        task("second", estimated_volume_m3=0.75)])
        result = solve_v5(data)
        self.assertEqual(len(result["routes"][0]["stops"]), 1)
        self.assertEqual(result["unassigned"][0]["reason"], "capacity_volume")
        data["vehicles"][0]["starting_volume_m3"] = 2.1
        result = solve_v5(data)
        self.assertEqual([item["reason"] for item in result["unassigned"]],
                         ["capacity_volume", "capacity_volume"])

    def test_contract_and_http_validation(self):
        data = payload([vehicle()], [task("a")])
        bad = copy.deepcopy(data)
        bad["contract_version"] = "v4"
        with app.test_client() as client:
            self.assertEqual(client.post("/solve_vrp", json=bad).status_code, 400)
            response = client.post("/solve_vrp", json=data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["status"], "success")


if __name__ == "__main__":
    unittest.main()
