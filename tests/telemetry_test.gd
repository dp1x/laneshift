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
	ok = ok and _test_failed_write_is_preserved()
	ok = ok and _test_queue_overflow_is_bounded_and_visible()
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

func _test_failed_write_is_preserved() -> bool:
	var logger = LoggerScript.new()
	logger.start_session()
	var valid_dir: String = logger.absolute_session_dir
	var blocker_path := valid_dir.path_join("not_a_directory")
	var blocker := FileAccess.open(blocker_path, FileAccess.WRITE)
	if blocker == null:
		logger.close()
		logger.free()
		return false
	blocker.store_string("blocker")
	blocker.close()
	logger.absolute_session_dir = blocker_path
	logger._write_batch({"filename": "events.jsonl", "records": [{"type": "recovered_test", "id": 701}]})
	logger.absolute_session_dir = valid_dir
	var recovered := false
	var recovery_paths: Array = []
	for property in logger.get_property_list():
		if property["name"] == "recovered_batches":
			recovered = int(logger.get("recovered_batches")) == 1
		if property["name"] == "recovery_files":
			recovery_paths = logger.get("recovery_files")
	var record_preserved := false
	for path in recovery_paths:
		var recovery_file := FileAccess.open(path, FileAccess.READ)
		if recovery_file == null:
			continue
		while not recovery_file.eof_reached():
			var line := recovery_file.get_line()
			if line.is_empty():
				continue
			var entry = JSON.parse_string(line)
			if typeof(entry) == TYPE_DICTIONARY and entry.get("record", {}).get("id", -1) == 701:
				record_preserved = true
		recovery_file.close()
	logger.close()
	logger.free()
	return recovered and record_preserved

func _test_queue_overflow_is_bounded_and_visible() -> bool:
	var logger = LoggerScript.new()
	logger.start_session()
	var has_capacity_setting := false
	for property in logger.get_property_list():
		if property["name"] == "max_queued_batches":
			has_capacity_setting = true
	if not has_capacity_setting:
		logger.close()
		logger.free()
		return false
	logger.max_queued_batches = 1
	logger.writer_mutex.lock()
	logger.write_queue.append({"filename": "events.jsonl", "records": []})
	logger.writer_mutex.unlock()
	for i in range(60):
		logger.log_telemetry({"episode_id": "overflow-test", "physics_tick": i})
	var bounded_and_reported: bool = bool(logger.get("degraded")) and int(logger.get("dropped_records")) == 60 and not bool(logger.get("enabled"))
	logger.close()
	logger.free()
	return bounded_and_reported
