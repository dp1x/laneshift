extends SceneTree

const LoggerScript = preload("res://scripts/telemetry/telemetry_logger.gd")

func _initialize() -> void:
	var logger = LoggerScript.new()
	logger.start_session()
	logger.log_telemetry({"episode_id": "test-episode", "physics_tick": 1})
	logger.log_event({"type": "test_event", "episode_id": "test-episode"})
	logger.log_episode({"episode_id": "test-episode", "termination_reason": "test"})
	logger.close()
	var ok := _contains_json_line(logger.session_dir + "/telemetry.jsonl", "physics_tick", 1)
	ok = ok and _contains_json_line(logger.session_dir + "/events.jsonl", "type", "test_event")
	ok = ok and _contains_json_line(logger.session_dir + "/episodes.jsonl", "termination_reason", "test")
	logger.free()
	if ok:
		print("TELEMETRY_TESTS_PASS")
		quit(0)
	else:
		push_error("telemetry logger must flush valid JSONL records before close returns")
		quit(1)

func _contains_json_line(path: String, key: String, expected: Variant) -> bool:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return false
	while not file.eof_reached():
		var parsed = JSON.parse_string(file.get_line())
		if typeof(parsed) == TYPE_DICTIONARY and parsed.get(key) == expected:
			file.close()
			return true
	file.close()
	return false
