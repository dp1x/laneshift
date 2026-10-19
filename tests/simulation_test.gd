extends SceneTree

const SimulationScript = preload("res://scripts/sim/simulation.gd")

var failures: Array[String] = []

func _initialize() -> void:
	_test_seed_reproduces_traffic_schedule()
	_test_action_changes_lane_and_braking_speed()
	_test_lane_reports_physical_lane_and_legal_actions()
	_test_collision_ends_episode_once()
	_test_episode_duration_ends_normally()
	_test_rejected_boundary_action_does_not_become_active()
	_test_manual_termination_is_summarized()
	if failures.is_empty():
		print("SIMULATION_TESTS_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _test_seed_reproduces_traffic_schedule() -> void:
	var a = SimulationScript.new()
	var b = SimulationScript.new()
	a.configure({"traffic_density": 0.8})
	b.configure({"traffic_density": 0.8})
	a.reset(554, {"traffic_density": 0.8})
	b.reset(554, {"traffic_density": 0.8})
	_check(a.traffic == b.traffic, "same seed must reproduce initial traffic")
	for i in range(240):
		a.step(1.0 / 60.0)
		b.step(1.0 / 60.0)
	_check(a.traffic == b.traffic, "same actions and ticks must reproduce traffic schedule")
	a.free()
	b.free()

func _test_action_changes_lane_and_braking_speed() -> void:
	var sim = SimulationScript.new()
	sim.reset(7)
	sim.traffic.clear()
	sim.next_spawn_z = 10000.0
	_check(sim.apply_action("LEFT", 0.4, "human"), "valid lane change should be accepted")
	var initial_x: float = sim.player_x
	sim.step(1.0 / 60.0)
	_check(sim.player_x < initial_x, "lane change should move player laterally")
	_check(sim.player_lane == 1 and sim.player_target_lane == 0, "current lane must remain physical until transition completes")
	var speed_before: float = sim.player_speed
	_check(sim.apply_action("BRAKE", 0.5, "human"), "brake action should be accepted")
	for i in range(30):
		sim.step(1.0 / 60.0)
	_check(sim.player_speed < speed_before, "braking should reduce player speed")
	sim.free()

func _test_lane_reports_physical_lane_and_legal_actions() -> void:
	var sim = SimulationScript.new()
	sim.reset(70)
	sim.traffic.clear()
	sim.next_spawn_z = 10000.0
	if sim.has_method("get_legal_actions"):
		_check(sim.get_legal_actions() == ["LEFT", "STAY", "RIGHT", "BRAKE"], "center lane should expose all four legal actions")
	else:
		_check(false, "simulation must expose authoritative legal actions")
	sim.apply_action("LEFT", 0.4, "human")
	if sim.has_method("get_legal_actions"):
		var transitioning_actions: Array = sim.get_legal_actions()
		_check("LEFT" not in transitioning_actions and "RIGHT" not in transitioning_actions, "steering must be unavailable during lane transition")
	for i in range(45):
		sim.step(1.0 / 60.0)
	var has_transition_field := false
	for property in sim.get_property_list():
		if property["name"] == "lane_change_in_progress":
			has_transition_field = true
	if has_transition_field:
		_check(sim.player_lane == 0 and not sim.get("lane_change_in_progress"), "lane should commit only after physical transition completes")
	else:
		_check(false, "simulation must expose lane transition state")
	if sim.has_method("get_legal_actions"):
		var left_edge_actions: Array = sim.get_legal_actions()
		_check("LEFT" not in left_edge_actions and "RIGHT" in left_edge_actions, "outer left lane must exclude only unavailable steering")
	sim.free()

func _test_collision_ends_episode_once() -> void:
	var sim = SimulationScript.new()
	sim.reset(8)
	var end_count := [0]
	sim.episode_ended.connect(func(_summary): end_count[0] += 1)
	sim.traffic = [{"id": "forced", "lane": 1, "z": 1.0, "speed": 0.0, "type": "car", "length": 4.5, "width": 1.8}]
	sim.step(1.0 / 60.0)
	_check(sim.status == "crashed", "overlapping traffic should crash the episode")
	_check(end_count[0] == 1, "collision should produce one episode summary")
	sim.step(1.0 / 60.0)
	_check(end_count[0] == 1, "ended episode must not emit duplicate summary")
	sim.free()

func _test_episode_duration_ends_normally() -> void:
	var sim = SimulationScript.new()
	sim.reset(9, {"episode_duration": 0.05})
	for i in range(4):
		sim.step(1.0 / 60.0)
	_check(sim.status == "finished", "episode duration should terminate a surviving run")
	sim.free()

func _test_rejected_boundary_action_does_not_become_active() -> void:
	var sim = SimulationScript.new()
	sim.reset(10)
	sim.player_lane = 0
	sim.player_target_lane = 0
	sim.player_x = sim.lane_center(0)
	_check(not sim.apply_action("LEFT", 0.2, "human"), "lane edge action should be rejected")
	_check(sim.current_action == "STAY", "rejected command must not appear as the active action")
	sim.free()

func _test_manual_termination_is_summarized() -> void:
	var sim = SimulationScript.new()
	sim.reset(11)
	var reasons: Array[String] = []
	sim.episode_ended.connect(func(summary): reasons.append(summary["termination_reason"]))
	if sim.has_method("finish_episode"):
		sim.finish_episode("menu_exit")
	else:
		_check(false, "simulation needs an explicit manual termination method")
	_check(reasons == ["menu_exit"], "manual termination should produce one named summary")
	sim.free()

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
