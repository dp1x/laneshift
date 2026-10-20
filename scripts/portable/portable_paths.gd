class_name PortablePaths
extends RefCounted

const DEV_ROOT_REL := ".portable-dev"

static func is_editor() -> bool:
	return OS.has_feature("editor")

static func get_portable_root() -> String:
	if is_editor():
		return ProjectSettings.globalize_path("res://" + DEV_ROOT_REL)
	return OS.get_executable_path().get_base_dir()

static func get_recordings_root() -> String:
	return get_portable_root().path_join("recordings")

static func get_logs_root() -> String:
	return get_portable_root().path_join("logs")

static func get_config_root() -> String:
	return get_portable_root().path_join("config")

static func get_transfer_root() -> String:
	return get_portable_root().path_join("transfer")

static func ensure_dir(path: String) -> bool:
	var err := DirAccess.make_dir_recursive_absolute(path)
	return err == OK

static func is_writable(path: String) -> bool:
	if not DirAccess.dir_exists_absolute(path):
		if not ensure_dir(path):
			return false
	var test_file := path.path_join(".write_test")
	var f := FileAccess.open(test_file, FileAccess.WRITE)
	if f == null:
		return false
	f.store_string("ok")
	f.close()
	DirAccess.remove_absolute(test_file)
	return true
