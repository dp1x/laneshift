extends Node
class_name TelemetryLogger

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
var writer_stopping: bool = false
var writer_error: String = ""
var writer_closed: bool = false

func start_session() -> void:
	var ts := Time.get_datetime_string_from_system().replace(":", "").replace("-", "").replace("T", "_")
	session_id = "session_%s_%03d" % [ts, int(Time.get_ticks_msec() % 1000)]
	session_dir = "user://runs/%s" % session_id
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

func log_telemetry(snapshot: Dictionary) -> void:
	if not enabled:
		return
	telemetry_buffer.append(snapshot)
	if telemetry_buffer.size() >= 60:
		_queue_records("telemetry.jsonl", telemetry_buffer)
		telemetry_buffer = []

func log_event(event: Dictionary) -> void:
	if not enabled:
		return
	events_buffer.append(event)
	if events_buffer.size() >= 64:
		_queue_records("events.jsonl", events_buffer)
		events_buffer = []

func log_episode(summary: Dictionary) -> void:
	if not enabled:
		return
	episodes_buffer.append(summary)
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

func _queue_records(filename: String, records: Array) -> void:
	if writer_closed or writer_thread == null or not writer_thread.is_started():
		return
	writer_mutex.lock()
	write_queue.append({"filename": filename, "records": records.duplicate(true)})
	writer_mutex.unlock()
	writer_semaphore.post()

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

func _write_batch(batch: Dictionary) -> void:
	var path := absolute_session_dir.path_join(String(batch["filename"]))
	var file := FileAccess.open(path, FileAccess.READ_WRITE)
	if file == null:
		file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		_set_writer_error("could not open log: %s" % path)
		return
	file.seek_end()
	for record in batch["records"]:
		file.store_line(JSON.stringify(record))
	var err := file.get_error()
	file.close()
	if err != OK:
		_set_writer_error("failed while writing log: %s" % path)

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
	if message != "":
		push_warning("LaneShift telemetry: " + message)

func close() -> void:
	if writer_closed:
		return
	flush()
	if writer_thread != null and writer_thread.is_started():
		writer_mutex.lock()
		writer_stopping = true
		writer_mutex.unlock()
		writer_semaphore.post()
		writer_thread.wait_to_finish()
	writer_closed = true
	_report_writer_error()

func _warn_write_failure(path: String, message: String) -> void:
	write_failures += 1
	if not warned_paths.has(path):
		warned_paths[path] = true
		push_warning("LaneShift: %s: %s" % [message, path])

func get_session_path() -> String:
	return ProjectSettings.globalize_path(session_dir) if session_dir != "" else ""
