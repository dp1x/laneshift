extends SceneTree

# Emits one JSON line per controlled traffic state. The Python audit tool
# (python/tools/audit_scene_observation.py) renders the same lines through the
# lane prompt and the scene prompt so engine, renderer and Kev can be compared.

const MainScene = preload("res://scenes/main.tscn")

const SCENARIOS := [
	{
		"name": "multiple_same_lane",
		"traffic": [
			{"id": "c_near", "lane": 1, "z": 110.0, "speed": 13.0, "type": "car"},
			{"id": "c_mid", "lane": 1, "z": 130.0, "speed": 12.0, "type": "truck"},
			{"id": "c_far", "lane": 1, "z": 150.0, "speed": 11.0, "type": "car"},
		],
	},
	{
		"name": "faster_farther",
		"traffic": [
			{"id": "near_slow", "lane": 1, "z": 110.0, "speed": 18.0, "type": "car"},
			{"id": "far_fast", "lane": 1, "z": 150.0, "speed": 10.0, "type": "car"},
		],
	},
	{
		"name": "directly_behind",
		"traffic": [
			{"id": "behind_fast", "lane": 1, "z": 87.0, "speed": 24.0, "type": "car"},
			{"id": "front_slow", "lane": 1, "z": 125.0, "speed": 12.0, "type": "car"},
		],
	},
	{
		"name": "adjacent_lanes",
		"traffic": [
			{"id": "left_ahead", "lane": 0, "z": 130.0, "speed": 14.0, "type": "car"},
			{"id": "right_ahead", "lane": 2, "z": 125.0, "speed": 13.0, "type": "truck"},
			{"id": "left_behind_off", "lane": 0, "z": 87.0, "speed": 22.0, "type": "car"},
		],
	},
	{
		"name": "off_screen_approaching",
		"traffic": [
			{"id": "off_left", "lane": 0, "z": 230.0, "speed": 16.0, "type": "car"},
			{"id": "off_right", "lane": 2, "z": 250.0, "speed": 15.0, "type": "truck"},
			{"id": "beyond", "lane": 1, "z": 400.0, "speed": 10.0, "type": "car"},
		],
	},
	{
		"name": "open_road",
		"traffic": [],
	},
	{
		"name": "blocked_current_lane",
		"traffic": [
			{"id": "front_block", "lane": 1, "z": 106.0, "speed": 10.0, "type": "car"},
			{"id": "left_clear", "lane": 0, "z": 135.0, "speed": 13.0, "type": "car"},
			{"id": "right_clear", "lane": 2, "z": 140.0, "speed": 12.0, "type": "car"},
		],
	},
	{
		"name": "both_adjacent_unsafe",
		"traffic": [
			{"id": "left_block", "lane": 0, "z": 108.0, "speed": 12.0, "type": "car"},
			{"id": "right_block", "lane": 2, "z": 109.0, "speed": 11.0, "type": "truck"},
			{"id": "center_clear", "lane": 1, "z": 160.0, "speed": 12.0, "type": "car"},
		],
	},
	{
		"name": "lane_safe_now_unsafe_soon",
		"traffic": [
			{"id": "right_closing", "lane": 2, "z": 120.0, "speed": 11.0, "type": "car"},
			{"id": "center_front", "lane": 1, "z": 135.0, "speed": 15.0, "type": "car"},
		],
	},
	{
		"name": "off_screen_behind",
		"traffic": [
			{"id": "behind_edge", "lane": 1, "z": 86.0, "speed": 24.0, "type": "car"},
			{"id": "front", "lane": 1, "z": 120.0, "speed": 12.0, "type": "car"},
		],
	},
]

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var game = MainScene.instantiate()
	root.add_child(game)
	await process_frame
	await process_frame
	game.start_game(7, {"traffic_density": 0.7}, "controlled_probe")
	for scenario in SCENARIOS:
		game.sim.player_z = 100.0
		game.sim.player_x = 0.0
		game.sim.player_lane = 1
		game.sim.player_target_lane = 1
		game.sim.lane_change_in_progress = false
		game.sim.lane_change_cooldown = 0.0
		game.sim.player_speed = 20.0
		game.sim.braking = false
		game.sim.current_action = "STAY"
		game.sim.traffic.clear()
		for spec in scenario["traffic"]:
			var length := 7.0 if spec["type"] == "truck" else 4.5
			var width := 2.4 if spec["type"] == "truck" else 1.8
			game.sim.traffic.append({
				"id": spec["id"], "lane": spec["lane"], "z": spec["z"],
				"speed": spec["speed"], "type": spec["type"],
				"length": length, "width": width,
			})
		var snapshot: Dictionary = game._observation_snapshot()
		snapshot["controlled_scenario"] = scenario["name"]
		print("SCENARIO_JSON:" + JSON.stringify(snapshot))
	game.free()
	quit(0)
