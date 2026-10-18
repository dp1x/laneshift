extends SceneTree

const MainScene = preload("res://scenes/main.tscn")

var failures: Array[String] = []

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var game = MainScene.instantiate()
	root.add_child(game)
	var start_button = game.menu_panel.find_child("Start", true, false)
	start_button.pressed.emit()
	_check(game.state == "playing", "title-screen Start button should enter gameplay")

	var press := InputEventKey.new()
	press.keycode = KEY_A
	press.physical_keycode = KEY_A
	press.pressed = true
	Input.parse_input_event(press)
	Input.flush_buffered_events()
	game._handle_human_input(1.0 / 60.0)
	_check(game.sim.player_target_lane == 0, "A key should request one smooth lane change")
	var release := InputEventKey.new()
	release.keycode = KEY_A
	release.physical_keycode = KEY_A
	release.pressed = false
	Input.parse_input_event(release)

	var observed_state: Dictionary = game.sim.get_snapshot()
	observed_state["physics_tick"] = 5
	observed_state["sim_time"] = 5.0 / 60.0
	observed_state["state_id"] = "%s:5" % game.sim.episode_id
	observed_state["player"]["speed"] = 19.25
	if game.has_method("_remember_decision_state"):
		game._remember_decision_state(observed_state)
	else:
		_check(false, "game should retain snapshots sent to the controller")
	game.sim.physics_tick = 18
	game.sim.sim_time = 18.0 / 60.0
	game.sim.lane_change_cooldown = 0.0
	game.sim.player_x = game.sim.lane_center(game.sim.player_lane)
	game.sim.player_target_lane = game.sim.player_lane
	game.external_control = true
	game._on_net_action("RIGHT", 0.35, 22, 5, game.sim.episode_id, "synthetic")
	var delayed_event: Dictionary = {}
	for event in game.telemetry.events_buffer:
		if event.get("type") == "action_decision" and event.get("controller_seq") == 22:
			delayed_event = event
	_check(not delayed_event.is_empty(), "delayed controller action should be recorded")
	_check(delayed_event.get("decision_tick", -1) == 5, "action record should preserve the controller state tick")
	_check(delayed_event.get("application_tick", -1) == 18, "action record should separate its application tick")
	_check(delayed_event.get("state_before", {}).get("player", {}).get("speed", -1.0) == 19.25, "action input should be the exact observed snapshot")
	game.external_control = false

	game.pause_game()
	_check(game.state == "paused" and game.pause_panel.visible, "pause should show a visible resume panel")
	game.resume_game()
	_check(game.state == "playing" and not game.pause_panel.visible, "resume should restore gameplay")

	game.sim.player_x = game.sim.lane_center(game.sim.player_lane)
	game.sim.traffic = [{"id": "forced-crash", "lane": game.sim.player_lane, "z": game.sim.player_z + 0.1, "speed": 0.0, "type": "car", "length": 4.5, "width": 1.8}]
	game.sim.step(1.0 / 60.0)
	_check(game.state == "gameover", "collision should show game-over state")
	_check(not game.gameover_result_label.text.is_empty(), "game-over screen should display the result")
	var restart_button = game.gameover_panel.find_child("Restart", true, false)
	restart_button.pressed.emit()
	_check(game.state == "playing" and game.sim.status == "running", "restart should begin a fresh episode")
	game.pause_game()
	var menu_button = game.pause_panel.find_child("Menu", true, false)
	menu_button.pressed.emit()
	_check(game.state == "menu" and game.sim.status == "finished", "return to menu should close the active episode")

	game.telemetry.close()
	game.free()
	if failures.is_empty():
		print("GAME_FLOW_TESTS_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
