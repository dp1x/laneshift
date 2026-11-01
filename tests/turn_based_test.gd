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
	game.turn_based = true
	game.evaluation_mode = true
	game.start_game(5, {"traffic_density": 0.0}, "turn_test")
	game.external_control = true
	game.net.connected_flag = true
	game.turn_waiting = true
	game.turn_wait_started_msec = Time.get_ticks_msec()
	var waiting_tick: int = game.sim.physics_tick
	for i in range(20):
		game._physics_process(1.0 / 60.0)
	_check(game.sim.physics_tick == waiting_tick, "simulation tick advanced while waiting for Kev")
	_check(game.turn_waiting, "turn waiter cleared without an action")

	var snapshot: Dictionary = game.sim.get_snapshot()
	game._remember_decision_state(snapshot)
	game.net.update_state(snapshot)
	game._on_net_action("LEFT", 0.2, 1, game.sim.physics_tick, game.sim.episode_id, "original_model", {})
	_check(not game.turn_waiting, "applied action did not resume the turn")
	_check(game.sim.current_action == "LEFT", "applied action was not LEFT")
	var resumed_tick: int = game.sim.physics_tick
	game._physics_process(1.0 / 60.0)
	_check(game.sim.physics_tick == resumed_tick + 1, "simulation did not resume after the action")

	game.free()
	if failures.is_empty():
		print("TURN_BASED_TEST_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
