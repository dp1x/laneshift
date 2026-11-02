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
	game.net.enabled = false
	game.net.connected_flag = true
	game.start_recording()
	# Keep events in the in-memory buffer for assertions instead of writing them
	# to disk every 0.25 s of simulated time.
	game.telemetry.flush_interval = 9999.0
	game.auto_run = true
	game.live_mode = true
	game.live_decision_interval = 1.0
	game.live_max_decision_age = 3.0
	game.live_request_timeout_msec = 20000
	game.live_inference_cooldown_msec = 0
	game.auto_total_episodes = 5
	game.start_game(5, {"traffic_density": 0.0}, "live_test")
	game.external_control = true
	game.live_decision_timer = game.live_decision_interval

	# 1. A request is sent without stopping the simulation.
	game._physics_process(1.0 / 60.0)
	_check(game.live_request_pending, "the first live decision should be requested")
	var first_tick: int = game.live_request_tick
	var first_seq: int = game.live_request_seq
	var tick_before: int = game.sim.physics_tick
	for i in range(120):
		game._physics_process(1.0 / 60.0)
	_check(game.sim.physics_tick == tick_before + 120, "the simulation must keep stepping while Kev thinks")
	_check(game.live_request_pending, "only one request may be in flight")
	_check(game.live_request_seq == first_seq, "no second request may start while one is pending")
	var request_events := _count_events(game, "live_decision_requested")
	if request_events != 1:
		var types: Array[String] = []
		for event in game.telemetry.events_buffer:
			types.append(String(event.get("type", "")))
		print("LIVE_DEBUG types=", types)
		print("LIVE_DEBUG request_seq=", game.live_request_seq, " pending=", game.live_request_pending,
			" recording=", game.recording_active, " enabled=", game.telemetry.enabled,
			" session=", game.telemetry.session_active, " dir=", game.telemetry.session_dir)
	_check(request_events == 1, "exactly one live request event should exist")

	# 2. A prompt answer is applied and carries its exact observation.
	game._on_net_action("LEFT", 0.35, 1, first_tick, game.sim.episode_id, "original_model", {
		"model_latency_ms": 120.0,
		"confidence": 0.9,
		"probabilities": {"LEFT": 0.9, "STAY": 0.1},
	})
	_check(not game.live_request_pending, "an applied answer should clear the request")
	_check(game.live_decisions_applied == 1, "an applied answer should be counted")
	_check(game.sim.player_target_lane == 0, "LEFT should move the target one lane left")
	var applied_event := _last_event(game, "action_decision")
	_check(int(applied_event.get("physics_tick", -1)) == first_tick, "the action record should keep the decision tick")
	_check(float(applied_event.get("decision_age_s", -1.0)) >= 1.9, "the action record should include observation age")

	# 3. A late answer is dropped and changes nothing.
	_wait_for_request(game, 120)
	var stale_tick: int = game.live_request_tick
	var target_before: int = game.sim.player_target_lane
	for i in range(int(game.live_max_decision_age * 60.0) + 30):
		game._physics_process(1.0 / 60.0)
	game._on_net_action("RIGHT", 0.35, 2, stale_tick, game.sim.episode_id, "original_model", {
		"model_latency_ms": 4000.0,
		"confidence": 0.8,
	})
	_check(game.live_decisions_stale == 1, "a late answer should be counted as stale")
	_check(not game.live_request_pending, "a stale answer should clear the request")
	_check(game.sim.player_target_lane == target_before, "a stale answer must not change the simulation")
	var stale_event := _last_event(game, "live_decision_dropped")
	_check(stale_event.get("reason", "") == "stale_observation", "the drop should be explicit")

	# 4. A request timeout keeps the episode alive.
	_wait_for_request(game, 240)
	game.live_request_started_msec = Time.get_ticks_msec() - game.live_request_timeout_msec - 1
	game._physics_process(1.0 / 60.0)
	_check(not game.live_request_pending, "a timed-out request should be cleared")
	_check(game.live_decision_timeouts == 1, "the timeout should be counted")
	_check(game.sim.status == "running", "a timeout must not end a live episode")
	_check(_count_events(game, "live_decision_timeout") == 1, "the timeout should be logged")

	# 5. A controller error also keeps the episode alive.
	_wait_for_request(game, 120)
	var error_tick: int = game.live_request_tick
	game._on_model_error("decision_timeout", game.sim.episode_id, error_tick)
	_check(not game.live_request_pending, "a model error should clear the request")
	_check(game.live_inference_errors == 1, "the model error should be counted")
	_check(game.sim.status == "running", "a model error must not end a live episode")
	_check(_count_events(game, "live_inference_error") == 1, "the model error should be logged")

	# 6. An answer with no matching request is ignored.
	var applied_before: int = game.live_decisions_applied
	game._on_net_action("STAY", 0.35, 9, error_tick, game.sim.episode_id, "original_model", {})
	_check(game.live_decisions_applied == applied_before, "an unmatched answer must not be applied")

	game.telemetry.close()
	game.free()
	if failures.is_empty():
		print("LIVE_DECISION_TESTS_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _wait_for_request(game, max_ticks: int) -> void:
	for i in range(max_ticks):
		if game.live_request_pending:
			return
		game._physics_process(1.0 / 60.0)
	if not game.live_request_pending:
		failures.append("expected a new decision request")

func _count_events(game, event_type: String) -> int:
	var count := 0
	for event in game.telemetry.events_buffer:
		if event.get("type") == event_type:
			count += 1
	return count

func _last_event(game, event_type: String) -> Dictionary:
	var found: Dictionary = {}
	for event in game.telemetry.events_buffer:
		if event.get("type") == event_type:
			found = event
	return found

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
