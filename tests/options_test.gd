extends SceneTree

const MainScript = preload("res://scripts/main.gd")

var failures: Array[String] = []

func _initialize() -> void:
	_test_option_values_survive_parsing()
	_test_defaults_are_sane()
	if failures.is_empty():
		print("OPTIONS_TESTS_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _test_option_values_survive_parsing() -> void:
	var args := PackedStringArray([
		"--auto-run",
		"--seeds=11,22,33",
		"--episodes=3",
		"--max-episodes=9",
		"--episode-duration=12.5",
		"--ws-port=8123",
		"--external-timeout-ms=40000",
		"--max-stale-ticks=6000",
		"--traffic-density=0.12",
		"--decision-history-s=240",
		"--live",
		"--cycle-seeds",
		"--decision-interval-s=2.5",
		"--max-decision-age-s=4.5",
		"--request-timeout-ms=21000",
		"--inference-cooldown-ms=100",
	])
	var options: Dictionary = MainScript.parse_options(args)
	_check(bool(options["auto_run"]), "auto-run flag should be recognised")
	_check(options["seeds"] == [11, 22, 33], "seeds should parse as a list of integers")
	_check(int(options["episodes"]) == 3, "episodes should parse")
	_check(int(options["max_episodes"]) == 9, "max-episodes should parse")
	_check(is_equal_approx(float(options["episode_duration"]), 12.5), "episode-duration should parse")
	_check(int(options["ws_port"]) == 8123, "ws-port should parse")
	_check(int(options["external_timeout_ms"]) == 40000, "external-timeout-ms should parse")
	_check(int(options["max_stale_ticks"]) == 6000, "max-stale-ticks should parse")
	_check(is_equal_approx(float(options["traffic_density"]), 0.12), "traffic-density should parse")
	_check(is_equal_approx(float(options["decision_history_s"]), 240.0), "decision-history-s should parse")
	_check(bool(options["live_mode"]), "live flag should be recognised")
	_check(bool(options["cycle_seeds"]), "cycle-seeds flag should be recognised")
	_check(is_equal_approx(float(options["live_decision_interval"]), 2.5), "live decision interval should parse")
	_check(is_equal_approx(float(options["live_max_decision_age"]), 4.5), "max decision age should parse")
	_check(int(options["live_request_timeout_ms"]) == 21000, "live request timeout should parse")
	_check(int(options["live_inference_cooldown_ms"]) == 100, "inference cooldown should parse")

func _test_defaults_are_sane() -> void:
	var options: Dictionary = MainScript.parse_options(PackedStringArray(["--auto-run"]))
	_check(int(options["ws_port"]) == 8766, "default websocket port should be 8766")
	_check(int(options["external_timeout_ms"]) == 1500, "default control timeout should be 1500 ms")
	_check(int(options["max_stale_ticks"]) == 30, "default staleness bound should be 30 ticks")
	_check(is_equal_approx(float(options["traffic_density"]), -1.0), "traffic density should be unset by default")
	_check(not bool(options["live_mode"]), "live mode should be off by default")
	_check(is_equal_approx(float(options["live_decision_interval"]), 1.0), "live decision interval should default to 1 s")
	_check(is_equal_approx(float(options["live_max_decision_age"]), 5.0), "max decision age should default to 5 s")

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
