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
	game.start_recording()
	game.start_game(11, {"traffic_density": 1.0}, "human_schema_test")
	await process_frame
	var accepted: bool = game._submit_action("LEFT", 0.2, "human", "human")
	_check(accepted, "human LEFT was rejected")
	await process_frame
	await process_frame
	game.stop_recording()
	await process_frame

	var session_dir: String = game.get_session_dir()
	_check(session_dir != "", "recording session directory missing")
	var events_path := session_dir.path_join("events.jsonl")
	var file := FileAccess.open(events_path, FileAccess.READ)
	_check(file != null, "events.jsonl missing")
	if file != null:
		var found := false
		while not file.eof_reached():
			var line := file.get_line().strip_edges()
			if line == "":
				continue
			var event = JSON.parse_string(line)
			if typeof(event) != TYPE_DICTIONARY or event.get("type") != "action_decision":
				continue
			found = true
			var state: Dictionary = event.get("state_before", {})
			_check(int(state.get("observation_schema", 0)) == 2, "human state_before schema is not 2")
			_check(typeof(state.get("traffic", null)) == TYPE_ARRAY, "human state_before traffic missing")
			_check(event.get("label_source") == "human", "human decision label_source is wrong")
			_check(event.get("human_action") == "LEFT", "human_action was not recorded")
		file.close()
		_check(found, "no action_decision event was written")
	game.free()
	_remove_dir(session_dir)
	if failures.is_empty():
		print("HUMAN_SCHEMA_TEST_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _remove_dir(path: String) -> void:
	if path == "":
		return
	var dir := DirAccess.open(path)
	if dir == null:
		return
	dir.list_dir_begin()
	var name := dir.get_next()
	while name != "":
		if not dir.current_is_dir():
			DirAccess.remove_absolute(path.path_join(name))
		name = dir.get_next()
	dir.list_dir_end()
	DirAccess.remove_absolute(path)

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
