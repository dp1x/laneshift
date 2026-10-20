extends Node
class_name Simulation

signal crashed(episode_id: String)
signal episode_ended(summary: Dictionary)
signal scenario_initialized(info: Dictionary)
signal vehicle_overtaken(vehicle_id: String)

# Lane index 0 is the leftmost lane as the driver sees it.
# The chase camera looks along +Z, so its screen-right vector is world -X;
# that puts the driver's left at positive X. Lane centers are therefore
# +4 (left), 0 (centre), -4 (right) metres in world X.
const LANE_CENTERS := [4.0, 0.0, -4.0]
const LANE_COUNT := 3

var seed_value: int = 1
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

var physics_tick: int = 0
var sim_time: float = 0.0

var player_lane: int = 1
var player_target_lane: int = 1
var player_x: float = 0.0
var player_z: float = 0.0
var player_velocity_x: float = 0.0
var player_speed: float = 20.0
var base_speed: float = 20.0
var max_speed: float = 28.0
var braking: bool = false
var brake_timer: float = 0.0
var action_timer: float = 0.0
var lane_change_cooldown: float = 0.0
var lane_change_in_progress: bool = false
var lane_change_speed: float = 15.0

var traffic: Array = []
var next_spawn_z: float = 60.0
var vehicle_id_counter: int = 0

var episode_id: String = ""
var status: String = "idle"

var score: float = 0.0
var distance: float = 0.0
var survival_time: float = 0.0
var collision_count: int = 0
var overtaken_count: int = 0
var action_counts: Dictionary = {}

var control_source: String = "human"
var current_action: String = "STAY"

var config: Dictionary = {}
var episode_duration: float = 0.0

var run_counter: int = 0

func configure(cfg: Dictionary) -> void:
	config = cfg.duplicate(true)
	base_speed = float(config.get("base_speed", 20.0))
	max_speed = float(config.get("max_speed", 28.0))
	episode_duration = float(config.get("episode_duration", 0.0))

func reset(scenario_seed: int, cfg: Dictionary = {}, requested_episode_id: String = "") -> void:
	if not cfg.is_empty():
		configure(cfg)
	seed_value = scenario_seed
	rng.seed = scenario_seed
	physics_tick = 0
	sim_time = 0.0
	player_lane = 1
	player_target_lane = 1
	player_x = LANE_CENTERS[1]
	player_z = 0.0
	player_velocity_x = 0.0
	player_speed = base_speed
	braking = false
	brake_timer = 0.0
	action_timer = 0.0
	lane_change_cooldown = 0.0
	lane_change_in_progress = false
	traffic.clear()
	vehicle_id_counter = 0
	score = 0.0
	distance = 0.0
	survival_time = 0.0
	collision_count = 0
	overtaken_count = 0
	action_counts.clear()
	control_source = "human"
	current_action = "STAY"
	status = "running"
	run_counter += 1
	episode_id = requested_episode_id if requested_episode_id != "" else "ep_%d_%d" % [scenario_seed, run_counter]
	next_spawn_z = player_z + 75.0
	_spawn_initial()
	emit_signal("scenario_initialized", {
		"seed": scenario_seed,
		"episode_id": episode_id,
		"config": config.duplicate(true),
		"initial_traffic": traffic.duplicate(true),
		"next_spawn_z": next_spawn_z
	})

func _spawn_initial() -> void:
	var density := float(config.get("traffic_density", 1.0))
	var n := int(round(float(rng.randi_range(2, 4)) * max(0.0, density)))
	for i in range(n):
		# The first vehicle is kept out of the player's lane so every episode
		# opens with a readable road rather than an unavoidable collision.
		var z := player_z + 75.0 + i * 34.0
		_spawn_vehicle(z, player_lane if i == 0 else -1)

## Spawns a vehicle. `avoid_lane` is skipped when set.
func _spawn_vehicle(zpos: float, avoid_lane: int = -1) -> void:
	var v := {}
	v["id"] = "veh_%d" % vehicle_id_counter
	vehicle_id_counter += 1
	var lane := int(rng.randi_range(0, LANE_COUNT - 1))
	if avoid_lane >= 0 and lane == avoid_lane:
		lane = (lane + 1 + int(rng.randi_range(0, LANE_COUNT - 2))) % LANE_COUNT
	v["lane"] = lane
	v["z"] = zpos + rng.randf_range(-4.0, 4.0)
	var tmin := float(config.get("traffic_speed_min", 8.0))
	var tmax := float(config.get("traffic_speed_max", 16.0))
	v["speed"] = rng.randf_range(tmin, tmax)
	var types := ["car", "car", "truck"]
	v["type"] = types[rng.randi() % types.size()]
	v["length"] = 4.5 if v["type"] == "car" else 7.0
	v["width"] = 1.8 if v["type"] == "car" else 2.4
	traffic.append(v)

func step(delta: float) -> void:
	if status != "running":
		return
	physics_tick += 1
	sim_time += delta

	if lane_change_cooldown > 0.0:
		lane_change_cooldown = max(0.0, lane_change_cooldown - delta)
	if action_timer > 0.0:
		action_timer -= delta
		if action_timer <= 0.0 and not braking:
			current_action = "STAY"

	if braking:
		brake_timer -= delta
		if brake_timer <= 0.0:
			braking = false
			if current_action == "BRAKE":
				current_action = "STAY"

	if braking:
		player_speed = max(base_speed * 0.45, player_speed - 14.0 * delta)
	else:
		player_speed = min(max_speed, player_speed + 3.0 * delta)

	var target_x := float(LANE_CENTERS[player_target_lane])
	var previous_x := player_x
	player_x = move_toward(player_x, target_x, lane_change_speed * delta)
	player_velocity_x = (player_x - previous_x) / delta if delta > 0.0 else 0.0

	if abs(player_x - target_x) < 0.05:
		player_lane = player_target_lane
		lane_change_in_progress = false

	var dz := player_speed * delta
	player_z += dz
	distance += dz
	survival_time += delta

	for v in traffic:
		v["z"] += v["speed"] * delta

	var density := float(config.get("traffic_density", 1.0))
	var interval_min := float(config.get("spawn_interval_min", 22.0))
	var interval_max := float(config.get("spawn_interval_max", 38.0))
	var lead := 150.0
	while next_spawn_z < player_z + lead:
		_spawn_vehicle(next_spawn_z)
		var iv: float = rng.randf_range(interval_min, interval_max) / max(0.1, density)
		next_spawn_z += iv

	var to_remove := []
	for v in traffic:
		if v["z"] < player_z - 15.0:
			to_remove.append(v)
	for v in to_remove:
		traffic.erase(v)
		overtaken_count += 1
		emit_signal("vehicle_overtaken", String(v["id"]))

	score = distance * 0.1 + survival_time * 5.0 + overtaken_count * 25.0

	for v in traffic:
		var gap := float(v["z"]) - player_z
		var traffic_x := float(LANE_CENTERS[int(v["lane"])])
		var lateral_extent := float(v.get("width", 1.8)) * 0.5 + 0.45
		var longitudinal_extent := float(v.get("length", 4.5)) * 0.5 + 0.8
		if abs(player_x - traffic_x) < lateral_extent and abs(gap) < longitudinal_extent:
			collision_count += 1
			status = "crashed"
			emit_signal("crashed", episode_id)
			_finalize_episode("crash")
			return

	if episode_duration > 0.0 and survival_time >= episode_duration:
		status = "finished"
		_finalize_episode("timeout")

func _finalize_episode(reason: String) -> void:
	var summary := {
		"episode_id": episode_id,
		"scenario_seed": seed_value,
		"score": score,
		"distance": distance,
		"survival_time": survival_time,
		"collision_count": collision_count,
		"overtaken_count": overtaken_count,
		"action_counts": action_counts.duplicate(true),
		"final_lane": player_lane,
		"termination_reason": reason,
		"physics_tick": physics_tick,
		"sim_time": sim_time,
		"control_source": control_source
	}
	emit_signal("episode_ended", summary)

func finish_episode(reason: String) -> void:
	if status != "running":
		return
	status = "finished"
	_finalize_episode(reason)

func apply_action(action: String, duration: float = 0.5, source: String = "human") -> bool:
	if action not in get_legal_actions():
		return false
	var accepted := false
	match action:
		"LEFT":
			player_target_lane = player_lane - 1
			lane_change_in_progress = true
			lane_change_cooldown = 0.08
			accepted = true
		"RIGHT":
			player_target_lane = player_lane + 1
			lane_change_in_progress = true
			lane_change_cooldown = 0.08
			accepted = true
		"STAY":
			accepted = true
			braking = false
			brake_timer = 0.0
		"BRAKE":
			braking = true
			brake_timer = max(0.1, duration)
			accepted = true
	if accepted:
		if current_action != action:
			action_counts[action] = int(action_counts.get(action, 0)) + 1
		current_action = action
		action_timer = max(0.1, duration)
		control_source = source
	return accepted

func get_legal_actions() -> Array:
	if status != "running":
		return []
	var legal: Array = []
	if not lane_change_in_progress and lane_change_cooldown <= 0.0 and player_lane > 0:
		legal.append("LEFT")
	legal.append("STAY")
	if not lane_change_in_progress and lane_change_cooldown <= 0.0 and player_lane < LANE_COUNT - 1:
		legal.append("RIGHT")
	legal.append("BRAKE")
	return legal

func get_snapshot() -> Dictionary:
	var tlist := []
	for v in traffic:
		var gap: float = float(v["z"]) - player_z
		tlist.append({
			"id": v["id"],
			"lane": v["lane"],
			"type": v["type"],
			"position": {"x": LANE_CENTERS[v["lane"]], "y": 0.0, "z": v["z"]},
			"velocity": {"x": 0.0, "y": 0.0, "z": v["speed"]},
			"relative_distance_z": gap,
			"relative_speed_z": player_speed - float(v["speed"])
		})
	return {
		"protocol_version": 1,
		"episode_id": episode_id,
		"scenario_seed": seed_value,
		"sim_time": sim_time,
		"state_id": "%s:%d" % [episode_id, physics_tick],
		"physics_tick": physics_tick,
		"player": {
			"position": {"x": player_x, "y": 0.0, "z": player_z},
			"lane": player_lane,
			"target_lane": player_target_lane,
			"lane_change_in_progress": lane_change_in_progress,
			"lane_change_cooldown": lane_change_cooldown,
			"speed": player_speed,
			"velocity": {"x": player_velocity_x, "y": 0.0, "z": player_speed},
			"current_action": current_action,
			"braking": braking
		},
		"legal_actions": get_legal_actions(),
		"traffic": tlist,
		"score": score,
		"distance": distance,
		"survival_time": survival_time,
		"collision_count": collision_count,
		"collision": status == "crashed",
		"episode_status": status,
		"control_source": control_source
	}

func lane_center(lane: int) -> float:
	return float(LANE_CENTERS[clamp(lane, 0, LANE_COUNT - 1)])
