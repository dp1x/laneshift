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

	game.start_game(1, {"traffic_density": 0.0})
	for i in range(20):
		game._physics_process(1.0 / 60.0)
	_check(game.telemetry.enabled, "playing before Record must not disable the telemetry logger")
	_check(not game.telemetry.session_active, "no session should exist before Record")

	game.start_recording()
	_check(game.recording_active, "Record should start a recording session")
	_check(game.telemetry.enabled, "recording logger should be enabled")
	_check(game.telemetry.session_active, "recording logger should have an active session")

	for i in range(30):
		game._physics_process(1.0 / 60.0)
	game._submit_action("LEFT", 0.2, "human", "human")
	for i in range(6):
		game._physics_process(1.0 / 60.0)
	var session_dir: String = game.telemetry.absolute_session_dir
	game.stop_recording()

	var telemetry_lines := _line_count(session_dir.path_join("telemetry.jsonl"))
	var event_lines := _line_count(session_dir.path_join("events.jsonl"))
	_check(telemetry_lines > 0, "telemetry.jsonl should contain snapshots after Record")
	_check(event_lines > 0, "events.jsonl should contain events after Record")

	var manifest := _read_json(session_dir.path_join("manifest.json"))
	_check(int(manifest.get("counts", {}).get("telemetry", 0)) == telemetry_lines, "manifest telemetry count should match the file")
	_check(int(manifest.get("counts", {}).get("events", 0)) == event_lines, "manifest event count should match the file")
	_check(String(manifest.get("session_id", "")) != "", "manifest should have a session id")

	_remove_tree(session_dir)
	game.free()
	if failures.is_empty():
		print("TELEMETRY_FLOW_TEST_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _line_count(path: String) -> int:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return 0
	var count := 0
	while not file.eof_reached():
		if not file.get_line().strip_edges().is_empty():
			count += 1
	file.close()
	return count

func _read_json(path: String) -> Dictionary:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return {}
	var parsed = JSON.parse_string(file.get_as_text())
	file.close()
	return parsed if typeof(parsed) == TYPE_DICTIONARY else {}

func _remove_tree(path: String) -> void:
	var dir := DirAccess.open(path)
	if dir != null:
		dir.list_dir_begin()
		var name := dir.get_next()
		while name != "":
			if name != "." and name != "..":
				var child := path.path_join(name)
				if dir.current_is_dir():
					_remove_tree(child)
				else:
					DirAccess.remove_absolute(child)
			name = dir.get_next()
		dir.list_dir_end()
	DirAccess.remove_absolute(path)

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
