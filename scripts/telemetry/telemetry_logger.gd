extends Node
class_name TelemetryLogger

const PortablePathsScript = preload("res://scripts/portable/portable_paths.gd")

var output_base_dir: String = ""
var build_commit: String = "dev"
var build_godot_version: String = ""
var schema_version: String = "1"
var telemetry_record_count: int = 0
var event_record_count: int = 0
var episode_record_count: int = 0
var utc_start_time: String = ""
var utc_end_time: String = ""
var session_status: String = "recording"
var session_dir: String = ""
var absolute_session_dir: String = ""
var session_id: String = ""
var telemetry_buffer: Array = []
var events_buffer: Array = []
var episodes_buffer: Array = []
var flush_interval: float = 1.0
var flush_timer: float = 0.0
var telemetry_count: int = 0
var enabled: bool = true
var warned_paths: Dictionary = {}
var write_failures: int = 0
var writer_thread: Thread
var writer_mutex := Mutex.new()
var writer_semaphore := Semaphore.new()
var write_queue: Array = []
var max_queued_batches: int = 128
var max_write_attempts: int = 3
var writer_stopping: bool = false
var writer_error: String = ""
var writer_closed: bool = false
var recovered_batches: int = 0
var unrecovered_batches: int = 0
var dropped_records: int = 0
var recovery_files: Array[String] = []
var degraded: bool = false

func start_session(base_dir: String = "") -> void:
	if base_dir == "":
		base_dir = PortablePathsScript.get_recordings_root()
	output_base_dir = base_dir
	DirAccess.make_dir_recursive_absolute(output_base_dir)
	var ts := Time.get_datetime_string_from_system().replace(":", "").replace("-", "").replace("T", "_")
	session_id = "session_%s_%03d" % [ts, int(Time.get_ticks_msec() % 1000)]
	session_dir = "%s/%s" % [output_base_dir, session_id]
	absolute_session_dir = ProjectSettings.globalize_path(session_dir)
	var err := DirAccess.make_dir_recursive_absolute(absolute_session_dir)
	if err != OK:
		_warn_write_failure(session_dir, "could not create telemetry session directory")
		enabled = false
		return
	writer_thread = Thread.new()
	var thread_error := writer_thread.start(_writer_loop)
	if thread_error != OK:
		_warn_write_failure(session_dir, "could not start telemetry writer thread")
		enabled = false
		return
	telemetry_buffer.clear()
	events_buffer.clear()
	episodes_buffer.clear()
	telemetry_count = 0
	telemetry_record_count = 0
	event_record_count = 0
	episode_record_count = 0
	utc_start_time = Time.get_datetime_string_from_system(true)
	utc_end_time = ""
	session_status = "recording"
	if build_godot_version == "":
		build_godot_version = "%d.%d.%d" % [Engine.get_version_info()["major"], Engine.get_version_info()["minor"], Engine.get_version_info()["patch"]]
	_write_manifest("recording")

func log_telemetry(snapshot: Dictionary) -> void:
	if not enabled:
		return
	telemetry_buffer.append(snapshot)
	telemetry_record_count += 1
	if telemetry_buffer.size() >= 60:
		_queue_records("telemetry.jsonl", telemetry_buffer)
		telemetry_buffer = []

func log_event(event: Dictionary) -> void:
	if not enabled:
		return
	events_buffer.append(event)
	event_record_count += 1
	if events_buffer.size() >= 64:
		_queue_records("events.jsonl", events_buffer)
		events_buffer = []

func log_episode(summary: Dictionary) -> void:
	if not enabled:
		return
	episodes_buffer.append(summary)
	episode_record_count += 1
	flush()

func tick(delta: float) -> void:
	flush_timer += delta
	if flush_timer >= flush_interval:
		flush_timer = 0.0
		flush()
		_report_writer_error()

func flush() -> void:
	if not telemetry_buffer.is_empty():
		_queue_records("telemetry.jsonl", telemetry_buffer)
		telemetry_buffer = []
	if not events_buffer.is_empty():
		_queue_records("events.jsonl", events_buffer)
		events_buffer = []
	if not episodes_buffer.is_empty():
		_queue_records("episodes.jsonl", episodes_buffer)
		episodes_buffer = []

func _queue_records(filename: String, records: Array) -> bool:
	if writer_closed or writer_thread == null or not writer_thread.is_started():
		_mark_queue_failure("telemetry writer is not available", records.size())
		return false
	writer_mutex.lock()
	if write_queue.size() >= max_queued_batches:
		writer_mutex.unlock()
		_mark_queue_failure("telemetry writer queue capacity reached", records.size())
		return false
	write_queue.append({"filename": filename, "records": records.duplicate(true)})
	writer_mutex.unlock()
	writer_semaphore.post()
	return true

func _writer_loop() -> void:
	while true:
		writer_semaphore.wait()
		writer_mutex.lock()
		var batch: Dictionary = {}
		if not write_queue.is_empty():
			batch = write_queue.pop_front()
		var should_stop := writer_stopping and write_queue.is_empty()
		writer_mutex.unlock()
		if not batch.is_empty():
			_write_batch(batch)
		if should_stop:
			return

func _write_batch(batch: Dictionary) -> bool:
	var path := absolute_session_dir.path_join(String(batch["filename"]))
	var last_error := "could not open or write log: %s" % path
	for attempt in range(max_write_attempts):
		var error_code: int = FileAccess.get_open_error()
		var file := FileAccess.open(path, FileAccess.READ_WRITE)
		if file == null:
			file = FileAccess.open(path, FileAccess.WRITE)
		if file != null:
			file.seek_end()
			for record in batch["records"]:
				file.store_line(JSON.stringify(record))
			error_code = file.get_error()
			file.close()
			if error_code == OK:
				return true
		last_error = "failed to write log: %s (error %s)" % [path, error_string(error_code)]
		_set_writer_error(last_error)
		if attempt + 1 < max_write_attempts:
			OS.delay_msec(50 * (1 << attempt))
	if _persist_recovery_batch(batch, last_error):
		writer_mutex.lock()
		recovered_batches += 1
		writer_error = "log write recovered to an emergency JSONL file: %s" % recovery_files[-1]
		writer_mutex.unlock()
		return true
	writer_mutex.lock()
	unrecovered_batches += 1
	degraded = true
	writer_error = "log write failed after retries and emergency recovery failed: %s" % last_error
	writer_mutex.unlock()
	return false

func _persist_recovery_batch(batch: Dictionary, original_error: String) -> bool:
	var recovery_dir := ProjectSettings.globalize_path("user://runs/_recovery/%s" % session_id)
	if DirAccess.make_dir_recursive_absolute(recovery_dir) != OK:
		return false
	var filename := String(batch["filename"]).get_basename() + ".jsonl"
	var path := recovery_dir.path_join(filename)
	var file := FileAccess.open(path, FileAccess.READ_WRITE)
	if file == null:
		file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		return false
	file.seek_end()
	for record in batch["records"]:
		file.store_line(JSON.stringify({
			"session_id": session_id,
			"source_file": batch["filename"],
			"write_error": original_error,
			"record": record
		}))
	var err := file.get_error()
	file.close()
	if err != OK:
		return false
	writer_mutex.lock()
	recovery_files.append(path)
	writer_mutex.unlock()
	return true

func _mark_queue_failure(message: String, record_count: int) -> void:
	writer_mutex.lock()
	write_failures += 1
	dropped_records += record_count
	degraded = true
	enabled = false
	writer_error = "%s; dropped %d buffered records and disabled logging" % [message, record_count]
	writer_mutex.unlock()
	_report_writer_error()

func _set_writer_error(message: String) -> void:
	writer_mutex.lock()
	writer_error = message
	write_failures += 1
	writer_mutex.unlock()

func _report_writer_error() -> void:
	writer_mutex.lock()
	var message := writer_error
	writer_error = ""
	writer_mutex.unlock()
	if message == "":
		return
	if warned_paths.has(message):
		return
	warned_paths[message] = true
	if is_inside_tree():
		push_warning("LaneShift telemetry: " + message)

func close() -> void:
	if writer_closed:
		return
	flush()
	utc_end_time = Time.get_datetime_string_from_system(true)
	session_status = "complete"
	_write_manifest("complete")
	if writer_thread != null and writer_thread.is_started():
		writer_mutex.lock()
		writer_stopping = true
		writer_mutex.unlock()
		writer_semaphore.post()
		writer_thread.wait_to_finish()
	writer_closed = true
	_report_writer_error()
	if unrecovered_batches > 0:
		push_error("LaneShift telemetry ended with %d unrecovered batches (%d dropped records)" % [unrecovered_batches, dropped_records])

func _warn_write_failure(path: String, message: String) -> void:
	write_failures += 1
	if not warned_paths.has(path):
		warned_paths[path] = true
		push_warning("LaneShift: %s: %s" % [message, path])

func _write_manifest(status: String) -> void:
	session_status = status
	if status == "complete":
		utc_end_time = Time.get_datetime_string_from_system(true)
	var manifest := {
		"session_id": session_id,
		"utc_start_time": utc_start_time,
		"utc_end_time": utc_end_time,
		"build_commit": build_commit,
		"engine_version": build_godot_version,
		"recording_schema_version": schema_version,
		"status": session_status,
		"counts": {
			"telemetry": telemetry_record_count,
			"events": event_record_count,
			"episodes": episode_record_count
		}
	}
	var path := absolute_session_dir.path_join("manifest.json")
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(manifest, "\t"))
		f.close()

func get_session_path() -> String:
	return ProjectSettings.globalize_path(session_dir) if session_dir != "" else ""
