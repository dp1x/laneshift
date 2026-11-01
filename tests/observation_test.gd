extends SceneTree

const MainScene = preload("res://scenes/main.tscn")

var failures: Array[String] = []

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var game = MainScene.instantiate()
	root.add_child(game)
	await process_frame
	await process_frame
	game.start_game(7, {"traffic_density": 1.0}, "obs_test")
	var snapshot: Dictionary = game._observation_snapshot()
	_check(int(snapshot.get("observation_schema", 0)) == 2, "observation schema is not 2")
	_check(snapshot.get("traffic", []).size() == game.sim.traffic.size(), "snapshot lost active traffic")
	_check(snapshot.get("player", {}).has("acceleration"), "player acceleration missing")
	_check(snapshot.get("player", {}).has("recent_actions"), "recent actions missing")
	_check(snapshot.get("player", {}).has("previous_action"), "previous action missing")
	_check_vehicles(game, snapshot)

	# Controlled states bypass spawning so each geometry is exact.
	_setup(game, [
		{"id": "same_near", "lane": 1, "z": 110.0, "speed": 15.0, "type": "car"},
		{"id": "same_far", "lane": 1, "z": 150.0, "speed": 12.0, "type": "truck"},
		{"id": "left_behind", "lane": 0, "z": 86.0, "speed": 24.0, "type": "car"},
		{"id": "right_on", "lane": 2, "z": 125.0, "speed": 18.0, "type": "car"},
		{"id": "ahead_off", "lane": 0, "z": 230.0, "speed": 11.0, "type": "car"},
		{"id": "way_ahead", "lane": 2, "z": 400.0, "speed": 19.0, "type": "car"},
	])
	snapshot = game._observation_snapshot()
	_check(snapshot["traffic"].size() == 6, "controlled state lost active traffic")
	_check_vehicles(game, snapshot)
	var on_screen := 0
	var off_screen := 0
	for vehicle in snapshot["traffic"]:
		if bool(vehicle["on_screen"]):
			on_screen += 1
		else:
			off_screen += 1
	_check(on_screen == 3, "expected 3 on-screen vehicles, got %d" % on_screen)
	_check(off_screen == 3, "expected 3 off-screen vehicles, got %d" % off_screen)
	var truck: Dictionary = {}
	for vehicle in snapshot["traffic"]:
		if vehicle["id"] == "same_far":
			truck = vehicle
	_check(is_equal_approx(float(truck.get("length", 0.0)), 7.0), "truck length missing")
	_check(is_equal_approx(float(truck.get("width", 0.0)), 2.4), "truck width missing")
	_check(is_equal_approx(float(truck.get("speed", 0.0)), 12.0), "absolute traffic speed missing")

	_setup(game, [])
	snapshot = game._observation_snapshot()
	_check(snapshot["traffic"].is_empty(), "open road lost its empty traffic")
	_check(snapshot["player"]["previous_action"] == "STAY", "open road previous action wrong")

	game.free()
	if failures.is_empty():
		print("OBSERVATION_TEST_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _setup(game, specs: Array) -> void:
	game.sim.player_z = 100.0
	game.sim.player_x = 0.0
	game.sim.player_lane = 1
	game.sim.player_target_lane = 1
	game.sim.lane_change_in_progress = false
	game.sim.lane_change_cooldown = 0.0
	game.sim.player_speed = 20.0
	game.sim.traffic.clear()
	for spec in specs:
		var length := 7.0 if spec["type"] == "truck" else 4.5
		var width := 2.4 if spec["type"] == "truck" else 1.8
		game.sim.traffic.append({
			"id": spec["id"],
			"lane": spec["lane"],
			"z": spec["z"],
			"speed": spec["speed"],
			"type": spec["type"],
			"length": length,
			"width": width,
		})

func _check_vehicles(game, snapshot: Dictionary) -> void:
	for vehicle in snapshot.get("traffic", []):
		var z_rel: float = float(vehicle.get("relative_distance_z", 0.0))
		var expected: bool = z_rel >= -12.0 and z_rel <= game.VIEW_Z
		_check(bool(vehicle.get("on_screen", false)) == expected, "on_screen does not match renderer cull")
		_check(bool(vehicle.get("ahead", false)) == (z_rel > 0.0), "ahead does not match signed z")
		_check(vehicle.has("length") and vehicle.has("width"), "collision dimensions missing")
		_check(vehicle.has("speed"), "absolute speed missing")
		_check(is_equal_approx(float(vehicle.get("absolute_speed", -1.0)), float(vehicle.get("speed", 0.0))), "absolute speed alias missing")

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
