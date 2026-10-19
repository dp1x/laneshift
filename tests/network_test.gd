extends SceneTree

const BridgeScript = preload("res://scripts/net/net_bridge.gd")

var failures: Array[String] = []
var action_count: int = 0
var rejected_reasons: Array[String] = []
var reset_count: int = 0
var last_metadata: Dictionary = {}

func _initialize() -> void:
	var bridge = BridgeScript.new()
	bridge.set_episode_id("ep-current")
	bridge.update_state({"episode_id": "ep-current", "physics_tick": 100})
	bridge.connected_flag = true
	bridge.action_received.connect(func(_action, _duration, _seq, _tick, _episode, _source, metadata):
		action_count += 1
		last_metadata = metadata
	)
	bridge.command_rejected.connect(func(_seq, reason, _episode, _tick, _metadata): rejected_reasons.append(reason))
	bridge.reset_received.connect(func(_seed, _config, _episode): reset_count += 1)
	bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 1, "tick": 82, "action": "LEFT", "duration": 0.4,
		"label_source": "rule_based", "decision_metadata": {"model_id": "fixture/run", "probabilities": {"LEFT": 0.8}}
	}))
	_check(action_count == 1, "recent delayed command should be accepted")
	_check(last_metadata.get("model_id", "") == "fixture/run", "model provenance should be forwarded with the action")
	_check(bridge.outbound_queue.back().get("status", "") == "validated", "transport ACK should mean validated, not simulation-applied")
	bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 2, "tick": 100, "action": "RIGHT"
	}))
	_check(action_count == 2, "current command after a delayed command should be accepted")
	bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 1, "tick": 100, "action": "RIGHT"
	}))
	_check(rejected_reasons.has("duplicate_or_stale_seq"), "duplicate sequence should be rejected")
	bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-old",
		"seq": 3, "tick": 100, "action": "RIGHT"
	}))
	_check(rejected_reasons.has("wrong_episode"), "previous episode command should be rejected")
	bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 3, "tick": 60, "action": "RIGHT"
	}))
	_check(rejected_reasons.has("stale_tick"), "stale tick command should be rejected")
	bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 3, "tick": 100, "action": "FLY"
	}))
	_check(rejected_reasons.has("invalid_action"), "unsupported action should be rejected")
	bridge._handle_message(JSON.stringify({
		"type": "reset", "protocol_version": 1, "seed": 42,
		"config": {"traffic_density": 0.75}, "episode_id": "ep-next"
	}))
	_check(reset_count == 1, "valid seeded reset should be accepted")
	bridge._handle_message(JSON.stringify({
		"type": "reset", "protocol_version": 1, "seed": 42,
		"config": {"unknown_setting": 10}
	}))
	_check(rejected_reasons.has("invalid_reset"), "unknown reset settings should be rejected")
	# A slow model may legitimately answer many ticks after the state it saw.
	var slow_bridge = BridgeScript.new()
	slow_bridge.set_episode_id("ep-current")
	slow_bridge.update_state({"episode_id": "ep-current", "physics_tick": 400})
	var slow_accepted := [0]
	slow_bridge.action_received.connect(func(_action, _duration, _seq, _tick, _episode, _source, _metadata): slow_accepted[0] += 1)
	slow_bridge.max_stale_ticks = 2000
	slow_bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 1, "tick": 40, "action": "BRAKE", "duration": 0.3
	}))
	_check(slow_accepted[0] == 1, "a slow model's command should be accepted within a widened staleness bound")
	slow_bridge.max_stale_ticks = 30
	slow_bridge._handle_message(JSON.stringify({
		"type": "action", "protocol_version": 1, "episode_id": "ep-current",
		"seq": 2, "tick": 40, "action": "BRAKE", "duration": 0.3
	}))
	_check(slow_accepted[0] == 1, "a command beyond the staleness bound should still be rejected")
	slow_bridge.free()
	bridge.free()
	if failures.is_empty():
		print("NETWORK_TESTS_PASS")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

func _check(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
