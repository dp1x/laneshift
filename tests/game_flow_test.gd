extends SceneTree

const MainScene = preload("res://scenes/main.tscn")

var failures: Array[String] = []

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var game = MainScene.instantiate()
	root.add_child(game)
	game.start_recording()
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

	# Place the controlled bike at the completed left-lane state used for this delayed decision.
	game.sim.player_lane = 0
	game.sim.player_target_lane = 0
	game.sim.player_x = game.sim.lane_center(0)
	game.sim.lane_change_in_progress = false
	game.sim.lane_change_cooldown = 0.0
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
	game.external_control = true
	game._on_net_action("RIGHT", 0.35, 22, 5, game.sim.episode_id, "synthetic", {})
	var delayed_event: Dictionary = {}
	for event in game.telemetry.events_buffer:
		if event.get("type") == "action_decision" and event.get("controller_seq") == 22:
			delayed_event = event
	_check(not delayed_event.is_empty(), "delayed controller action should be recorded")
	_check(delayed_event.get("decision_tick", -1) == 5, "action record should preserve the controller state tick")
	_check(delayed_event.get("application_tick", -1) == 18, "action record should separate its application tick")
	_check(delayed_event.get("state_before", {}).get("player", {}).get("speed", -1.0) == 19.25, "action input should be the exact observed snapshot")
	_check(_has_event(game.telemetry.events_buffer, "external_action_received", 22), "external request receipt should be logged before action outcome")
	var blocked_state: Dictionary = game.sim.get_snapshot()
	blocked_state["physics_tick"] = 6
	blocked_state["state_id"] = "%s:6" % game.sim.episode_id
	blocked_state["legal_actions"] = ["STAY", "BRAKE"]
	game._remember_decision_state(blocked_state)
	game.sim.physics_tick = 19
	game.sim.player_lane = 1
	game.sim.player_target_lane = 1
	game.sim.player_x = game.sim.lane_center(1)
	game.sim.lane_change_in_progress = false
	game.sim.lane_change_cooldown = 0.0
	game.sim.current_action = "STAY"
	game._on_net_action("LEFT", 0.35, 23, 6, game.sim.episode_id, "synthetic", {})
	_check(game.sim.player_target_lane == 1, "command illegal in observed state must not become legal by arrival time")
	var illegal_event_found := false
	for event in game.telemetry.events_buffer:
		if event.get("type") == "external_action_rejected" and event.get("controller_seq") == 23:
			illegal_event_found = true
	_check(illegal_event_found, "illegal decision-time action should have a rejection event")
	game.external_control = false

	game.pause_game()
	_check(game.state == "paused" and game.pause_panel.visible, "pause should show a visible resume panel")
	_check(_has_event(game.telemetry.events_buffer, "pause", -1), "pause transition should be logged")
	game.resume_game()
	_check(game.state == "playing" and not game.pause_panel.visible, "resume should restore gameplay")
	_check(_has_event(game.telemetry.events_buffer, "resume", -1), "resume transition should be logged")

	game.sim.player_x = game.sim.lane_center(game.sim.player_lane)
	game.sim.traffic = [{"id": "forced-crash", "lane": game.sim.player_lane, "z": game.sim.player_z + 0.1, "speed": 0.0, "type": "car", "length": 4.5, "width": 1.8}]
	game.sim.step(1.0 / 60.0)
	_check(game.state == "gameover", "collision should show game-over state")
	_check(not game.gameover_result_label.text.is_empty(), "game-over screen should display the result")
	var restart_button = game.gameover_panel.find_child("Restart", true, false)
	restart_button.pressed.emit()
	_check(game.state == "playing" and game.sim.status == "running", "restart should begin a fresh episode")
	_check(_has_event(game.telemetry.events_buffer, "episode_reset", -1), "restart should emit an explicit reset event")
	game.pause_game()
	var menu_button = game.pause_panel.find_child("Menu", true, false)
	menu_button.pressed.emit()
	_check(game.state == "menu" and game.sim.status == "finished", "return to menu should close the active episode")

	game.auto_run = true
	game.start_game(314, {"episode_duration": 0.1})
	var terminal_episode: String = game.sim.episode_id
	game.sim.finish_episode("timeout")
	# Auto-run waits briefly between episodes; exercise physics callbacks during that terminal window.
	game.state = "playing"
	for i in range(8):
		game._physics_process(1.0 / 30.0)

	game.sim.finish_episode("timeout")

	game.telemetry.close()
	var telemetry_file := FileAccess.open(game.telemetry.session_dir + "/telemetry.jsonl", FileAccess.READ)
	var terminal_snapshot_count := 0
	if telemetry_file != null:
		while not telemetry_file.eof_reached():
			var line := telemetry_file.get_line()
			if line.is_empty():
				continue
			var record = JSON.parse_string(line)
			if typeof(record) == TYPE_DICTIONARY and record.get("state_id", "").begins_with(terminal_episode + ":") and record.get("episode_status") != "running":
				terminal_snapshot_count += 1
		telemetry_file.close()
	_check(terminal_snapshot_count == 1, "ended episode should persist exactly one terminal state, never repeated terminal ticks")
	_check(_steering_matches_screen(game), "steering must move the player in the direction the player sees")
	game.free()
	if failures.is_empty():
		print("GAME_FLOW_TESTS_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _steering_matches_screen(game) -> bool:
	game.sim.reset(4242, {"traffic_density": 0.0})
	game.sim.traffic.clear()
	game.sim.next_spawn_z = 100000.0
	var before: float = game._world_to_screen_x(game.sim.get_snapshot()["player"]["position"]["x"])
	game.sim.apply_action("LEFT", 0.4, "human")
	for i in range(40):
		game.sim.step(1.0 / 60.0)
	var after: float = game._world_to_screen_x(game.sim.get_snapshot()["player"]["position"]["x"])
	return after < before

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)

func _has_event(events: Array, event_type: String, controller_seq: int) -> bool:
	for event in events:
		if event.get("type") == event_type and (controller_seq < 0 or event.get("controller_seq", -1) == controller_seq):
			return true
	return false
