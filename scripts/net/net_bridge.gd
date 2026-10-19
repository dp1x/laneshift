extends Node
class_name NetBridge

signal connected()
signal disconnected()
signal action_received(action: String, duration: float, seq: int, tick: int, episode_id: String, label_source: String, decision_metadata: Dictionary)
signal reset_received(seed: int, config: Dictionary, episode_id: String)
signal command_rejected(seq: int, reason: String, episode_id: String, tick: int, decision_metadata: Dictionary)

var url: String = "ws://127.0.0.1:8765"
var socket: WebSocketPeer = WebSocketPeer.new()
var connected_flag: bool = false
var last_seq: int = -1
var current_episode_id: String = ""
var last_tick: int = -1
var current_state_tick: int = -1
var max_stale_ticks: int = 30
var last_action_received_msec: int = 0
var reconnect_timer: float = 0.0
var reconnect_interval: float = 3.0
var try_reconnect: bool = true
var outbound_queue: Array = []

func set_episode_id(eid: String) -> void:
	current_episode_id = eid
	last_seq = -1
	last_tick = -1
	current_state_tick = -1
	var retained_events: Array = []
	for message in outbound_queue:
		if message.get("type", "") == "event":
			retained_events.append(message)
	outbound_queue = retained_events

func update_state(snapshot: Dictionary) -> void:
	var state_episode := String(snapshot.get("episode_id", ""))
	if state_episode != current_episode_id:
		set_episode_id(state_episode)
	current_state_tick = int(snapshot.get("physics_tick", -1))

func _ready() -> void:
	_connect_socket()

func _connect_socket() -> void:
	var err := socket.connect_to_url(url)
	if err != OK:
		connected_flag = false

func _process(delta: float) -> void:
	socket.poll()
	var state := socket.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		if not connected_flag:
			connected_flag = true
			emit_signal("connected")
			_send({"type": "hello", "protocol_version": 1, "client": "godot"})
		# receive
		while socket.get_available_packet_count() > 0:
			var pkt := socket.get_packet()
			var text := pkt.get_string_from_utf8()
			_handle_message(text)
		# send queued
		while not outbound_queue.is_empty():
			var msg = outbound_queue.pop_front()
			socket.send_text(JSON.stringify(msg))
	elif state == WebSocketPeer.STATE_CLOSED:
		if connected_flag:
			connected_flag = false
			outbound_queue.clear()
			emit_signal("disconnected")
		if try_reconnect:
			reconnect_timer += delta
			if reconnect_timer >= reconnect_interval:
				reconnect_timer = 0.0
				socket = WebSocketPeer.new()
				_connect_socket()
	else:
		reconnect_timer = 0.0

func send_state(snapshot: Dictionary) -> void:
	if not connected_flag:
		return
	var msg := {"type": "state", "protocol_version": 1}
	for k in snapshot.keys():
		msg[k] = snapshot[k]
	outbound_queue.append(msg)

func send_event(event: Dictionary) -> void:
	if not connected_flag:
		return
	outbound_queue.append({"type": "event", "protocol_version": 1, "event": event})

func _handle_message(text: String) -> void:
	var parsed = JSON.parse_string(text)
	if parsed == null or typeof(parsed) != TYPE_DICTIONARY:
		return
	var msg: Dictionary = parsed
	if not _is_integer_number(msg.get("protocol_version")) or int(msg["protocol_version"]) != 1:
		return
	var mtype := String(msg.get("type", ""))
	match mtype:
		"action":
			var action := String(msg.get("action", ""))
			if action not in ["LEFT", "STAY", "RIGHT", "BRAKE"]:
				_send_reject("invalid_action", msg)
				return
			var ep := String(msg.get("episode_id", ""))
			if ep == "" or ep != current_episode_id:
				_send_reject("wrong_episode", msg)
				return
			if not _is_integer_number(msg.get("seq")) or not _is_integer_number(msg.get("tick")):
				_send_reject("invalid_sequence_or_tick", msg)
				return
			var seq := int(msg["seq"])
			if seq < 0 or seq <= last_seq:
				_send_reject("duplicate_or_stale_seq", msg)
				return
			var tick := int(msg["tick"])
			if current_state_tick < 0 or tick < current_state_tick - max_stale_ticks or tick > current_state_tick + 2 or tick < last_tick:
				_send_reject("stale_tick", msg)
				return
			var duration_value = msg.get("duration", 0.35)
			if typeof(duration_value) not in [TYPE_INT, TYPE_FLOAT]:
				_send_reject("invalid_duration", msg)
				return
			var dur := float(duration_value)
			if not is_finite(dur) or dur <= 0.0 or dur > 5.0:
				_send_reject("invalid_duration", msg)
				return
			var decision_metadata: Variant = msg.get("decision_metadata", {})
			if typeof(decision_metadata) != TYPE_DICTIONARY:
				_send_reject("invalid_decision_metadata", msg)
				return
			last_seq = max(last_seq, seq)
			last_tick = max(last_tick, tick)
			last_action_received_msec = Time.get_ticks_msec()
			var label_source := String(msg.get("label_source", "external_ai"))
			if label_source not in ["human", "rule_based", "original_model", "fine_tuned_model", "synthetic", "external_ai", "controller_fallback"]:
				label_source = "external_ai"
			emit_signal("action_received", action, dur, seq, tick, ep, label_source, decision_metadata)
			_send({"type": "ack", "protocol_version": 1, "seq": seq, "status": "validated"})
		"reset":
			if not _valid_reset_message(msg):
				_send_reject("invalid_reset", msg)
				return
			var seed := int(msg["seed"])
			var cfg: Dictionary = msg.get("config", {})
			var ep := String(msg.get("episode_id", ""))
			emit_signal("reset_received", seed, cfg, ep)
			_send({"type": "ack", "protocol_version": 1, "status": "reset_accepted", "episode_id": ep})
		"ping":
			_send({"type": "pong", "protocol_version": 1})

func _valid_reset_message(msg: Dictionary) -> bool:
	if not _is_integer_number(msg.get("seed")):
		return false
	var cfg = msg.get("config", {})
	if typeof(cfg) != TYPE_DICTIONARY:
		return false
	var limits := {
		"traffic_density": Vector2(0.1, 5.0),
		"traffic_speed_min": Vector2(0.0, 50.0),
		"traffic_speed_max": Vector2(0.0, 50.0),
		"spawn_interval_min": Vector2(1.0, 1000.0),
		"spawn_interval_max": Vector2(1.0, 1000.0),
		"episode_duration": Vector2(0.0, 3600.0),
		"base_speed": Vector2(1.0, 100.0),
		"max_speed": Vector2(1.0, 120.0)
	}
	var effective := {
		"traffic_speed_min": 8.0,
		"traffic_speed_max": 16.0,
		"spawn_interval_min": 22.0,
		"spawn_interval_max": 38.0,
		"base_speed": 20.0,
		"max_speed": 28.0
	}
	for key in cfg:
		if not limits.has(key) or typeof(cfg[key]) not in [TYPE_INT, TYPE_FLOAT]:
			return false
		var value := float(cfg[key])
		if not is_finite(value) or value < limits[key].x or value > limits[key].y:
			return false
		effective[key] = value
	if effective["traffic_speed_min"] > effective["traffic_speed_max"]:
		return false
	if effective["spawn_interval_min"] > effective["spawn_interval_max"]:
		return false
	if effective["base_speed"] > effective["max_speed"]:
		return false
	var episode: Variant = msg.get("episode_id", "")
	return typeof(episode) == TYPE_STRING and episode.length() <= 128

func _is_integer_number(value: Variant) -> bool:
	if typeof(value) not in [TYPE_INT, TYPE_FLOAT]:
		return false
	var number := float(value)
	return is_finite(number) and number == floor(number)

func _send(msg: Dictionary) -> void:
	if connected_flag:
		outbound_queue.append(msg)

func _send_reject(reason: String, orig: Dictionary) -> void:
	var seq := int(orig.get("seq", -1))
	var ep := String(orig.get("episode_id", ""))
	var tick := int(orig.get("tick", -1))
	var metadata: Variant = orig.get("decision_metadata", {})
	if typeof(metadata) != TYPE_DICTIONARY:
		metadata = {}
	emit_signal("command_rejected", seq, reason, ep, tick, metadata)
	outbound_queue.append({"type": "reject", "protocol_version": 1, "seq": seq, "reason": reason})
