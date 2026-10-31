extends Node2D

const PortablePathsScript = preload("res://scripts/portable/portable_paths.gd")

const DEFAULT_CONFIG := {
	"traffic_density": 0.7,
	"traffic_speed_min": 13.0,
	"traffic_speed_max": 21.0,
	"spawn_interval_min": 55.0,
	"spawn_interval_max": 95.0,
	"episode_duration": 0.0,
	"base_speed": 20.0,
	"max_speed": 28.0
}

# 2D presentation constants. World units are metres. The road uses one
# pseudo-perspective projection: lane centres are 4 m apart at the player and
# converge toward a horizon. Every sprite is sized with the same metres-per-pixel
# scale as the road, so lanes, vehicles and collision extents agree.
const LANE_PX := 120.0
const LANE_HALF_PX := 60.0
const ROAD_HALF_PX := 180.0
const PX_PER_M_NEAR := 30.0
const VIEW_Z := 110.0
const CAMERA_DISTANCE := 55.0
const HORIZON_Y := 0.14
const DASH_LENGTH_M := 2.6
const DASH_GAP_M := 2.4
const PLAYER_SCREEN_Y := 0.80
const BIKE_WIDTH_M := 0.9
const BIKE_LENGTH_M := 2.2

var sim: Simulation
var telemetry: TelemetryLogger
var net: NetBridge

var ui_layer: CanvasLayer
var hud: Control
var menu_panel: Control
var pause_panel: Control
var gameover_panel: Control

var state: String = "menu" # menu, playing, paused, gameover
var score_label: Label
var dist_label: Label
var speed_label: Label
var best_label: Label
var status_label: Label
var conn_label: Label
var msg_label: Label
var gameover_result_label: Label
var record_button: Button
var record_status_label: Label
var notice_label: Label
var countdown_label: Label
var menu_backdrop: ColorRect
var ai_panel: PanelContainer
var ai_model_label: Label
var ai_action_label: Label
var ai_latency_label: Label
var notice_timer: float = 0.0
var menu_scroll: float = 0.0

var best_score: float = 0.0
var highscore_path: String = ""

var telemetry_timer: float = 0.0
var telemetry_hz: float = 30.0
var decision_timer: float = 0.0
var decision_hz: float = 10.0

var control_mode: String = "human" # human, auto
var external_control: bool = false

var auto_run: bool = false
var record_on_start: bool = false
var take_control_on_start: bool = false
var auto_seeds: Array = [1, 2, 3]
var auto_episodes_per_seed: int = 1
var auto_episode_index: int = 0
var auto_total_episodes: int = 0
var auto_max_episodes: int = 0
var auto_running: bool = false
var auto_episode_duration: float = 0.0
var sim_speed_scale: float = 1.0
var countdown_seconds: float = 0.0
var countdown_timer: float = 0.0
var no_controller: bool = false
var start_fullscreen: bool = true
var action_event_sequence: int = 0
var external_timeout_msec: int = 1500
var max_stale_ticks: int = 30
var traffic_density_override: float = -1.0
var last_external_action_msec: int = 0
var websocket_port: int = 8766
var decision_state_history: Dictionary = {}
var decision_state_history_seconds: float = 180.0
var last_score_milestone: int = 0

var build_commit: String = "dev"
var build_godot_version: String = ""
var shake_timer: float = 0.0
var shake_intensity: float = 0.0
var shake_offset: Vector2 = Vector2.ZERO
var crash_hold: float = 0.0
var crash_flash: float = 0.0
var engine_player: AudioStreamPlayer
var sfx_player: AudioStreamPlayer

var recording_active: bool = false
var recording_session_id: String = ""
var recording_start_msec: int = 0
var recorded_event_count: int = 0
var recorded_sample_count: int = 0
var recorded_episode_count: int = 0
var brake_held: bool = false

var rng_visual: RandomNumberGenerator = RandomNumberGenerator.new()

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	rng_visual.randomize()
	PortablePathsScript.ensure_dir(PortablePathsScript.get_config_root())
	highscore_path = PortablePathsScript.get_config_root().path_join("highscore.save")
	_load_highscore()
	_parse_args()
	sim = Simulation.new()
	add_child(sim)
	sim.configure(DEFAULT_CONFIG.duplicate())
	sim.crashed.connect(_on_crashed)
	sim.episode_ended.connect(_on_episode_ended)
	sim.scenario_initialized.connect(_on_scenario_init)
	sim.vehicle_overtaken.connect(_on_vehicle_overtaken)
	_build_ui()
	_build_audio()
	telemetry = TelemetryLogger.new()
	add_child(telemetry)
	net = NetBridge.new()
	net.enabled = not no_controller
	net.url = "ws://127.0.0.1:%d" % websocket_port
	net.max_stale_ticks = max_stale_ticks
	add_child(net)
	net.action_received.connect(_on_net_action)
	net.reset_received.connect(_on_net_reset)
	net.command_rejected.connect(_on_command_rejected)
	net.connected.connect(_on_net_connected)
	net.disconnected.connect(_on_net_disconnected)
	if auto_run:
		_start_auto_run()
	elif record_on_start:
		start_recording()
	if DisplayServer.get_name() != "headless":
		if start_fullscreen:
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
		else:
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MAXIMIZED)
	queue_redraw()

func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		args = OS.get_cmdline_args()
	var options := parse_options(args)
	auto_run = bool(options["auto_run"])
	record_on_start = bool(options["record_on_start"])
	take_control_on_start = bool(options["take_control_on_start"])
	auto_seeds = options["seeds"]
	auto_episodes_per_seed = int(options["episodes"])
	auto_max_episodes = int(options["max_episodes"])
	auto_episode_duration = float(options["episode_duration"])
	sim_speed_scale = float(options["sim_speed_scale"])
	countdown_seconds = float(options["countdown"])
	no_controller = bool(options["no_controller"])
	start_fullscreen = bool(options["fullscreen"])
	websocket_port = int(options["ws_port"])
	external_timeout_msec = int(options["external_timeout_ms"])
	max_stale_ticks = int(options["max_stale_ticks"])
	traffic_density_override = float(options["traffic_density"])
	decision_state_history_seconds = float(options["decision_history_s"])
	if auto_seeds.is_empty():
		auto_seeds = [1]
	auto_episodes_per_seed = max(1, auto_episodes_per_seed)

static func parse_options(args: PackedStringArray) -> Dictionary:
	var options := {
		"auto_run": false,
		"record_on_start": false,
		"take_control_on_start": false,
		"seeds": [],
		"episodes": 1,
		"max_episodes": 0,
		"episode_duration": 0.0,
		"sim_speed_scale": 1.0,
		"countdown": 0.0,
		"no_controller": false,
		"fullscreen": true,
		"ws_port": 8766,
		"external_timeout_ms": 1500,
		"max_stale_ticks": 30,
		"traffic_density": -1.0,
		"decision_history_s": 180.0
	}
	for arg in args:
		if arg == "--auto-run":
			options["auto_run"] = true
			continue
		if arg == "--record-on-start":
			options["record_on_start"] = true
			continue
		if arg == "--take-control-on-start":
			options["take_control_on_start"] = true
			continue
		if arg == "--no-controller":
			options["no_controller"] = true
			continue
		if arg == "--fullscreen":
			options["fullscreen"] = true
			continue
		if arg == "--windowed":
			options["fullscreen"] = false
			continue
		var split := arg.split("=", true, 1)
		if split.size() != 2:
			continue
		var key := split[0]
		var value := split[1]
		match key:
			"--seeds":
				var seeds: Array = []
				for part in value.split(",", false):
					seeds.append(int(part))
				options["seeds"] = seeds
			"--episodes":
				options["episodes"] = int(value)
			"--max-episodes":
				options["max_episodes"] = int(value)
			"--episode-duration":
				options["episode_duration"] = float(value)
			"--sim-speed-scale":
				options["sim_speed_scale"] = clampf(float(value), 0.1, 4.0)
			"--countdown":
				options["countdown"] = clampf(float(value), 0.0, 10.0)
			"--ws-port":
				options["ws_port"] = clampi(int(value), 1, 65535)
			"--external-timeout-ms":
				options["external_timeout_ms"] = maxi(int(value), 100)
			"--max-stale-ticks":
				options["max_stale_ticks"] = maxi(int(value), 1)
			"--traffic-density":
				options["traffic_density"] = maxf(float(value), 0.0)
			"--decision-history-s":
				options["decision_history_s"] = maxf(float(value), 1.0)
	return options

# ---------------------------------------------------------------------------
# 2D rendering
# ---------------------------------------------------------------------------

func _world_to_screen_x(world_x: float) -> float:
	return _screen_x_for_world(world_x, 1.0)

func _screen_x_for_world(world_x: float, scale: float) -> float:
	var center_x := get_viewport_rect().size.x * 0.5
	return center_x - (world_x / 4.0) * LANE_PX * scale

func _project_z(z_rel: float) -> Dictionary:
	var view := get_viewport_rect().size
	var horizon := view.y * HORIZON_Y
	var player_y := view.y * PLAYER_SCREEN_Y
	var depth := 1.0 + maxf(z_rel, -12.0) / CAMERA_DISTANCE
	var scale := 1.0 / depth
	return {"y": horizon + (player_y - horizon) * scale, "scale": scale}

func _draw() -> void:
	if sim == null:
		return
	var view := get_viewport_rect().size
	var center_x := view.x * 0.5
	var player_y := view.y * PLAYER_SCREEN_Y
	var horizon := view.y * HORIZON_Y
	var scroll_z := sim.player_z if state != "menu" else menu_scroll

	draw_set_transform(shake_offset, 0.0, Vector2.ONE)

	# night sky, horizon glow and dark ground
	draw_rect(Rect2(0, 0, view.x, horizon), Color(0.012, 0.02, 0.04))
	draw_rect(Rect2(0, horizon, view.x, view.y - horizon), Color(0.02, 0.048, 0.04))
	for i in range(5):
		var t := float(i) / 4.0
		var glow_y := horizon - 22.0 + t * 44.0
		var glow_alpha := 0.10 * (1.0 - absf(t - 0.45) * 1.7)
		draw_rect(Rect2(0, glow_y, view.x, 8), Color(0.05, 0.16, 0.2, maxf(0.0, glow_alpha)))
	draw_rect(Rect2(0, horizon - 2, view.x, 3), Color(0.12, 0.3, 0.34, 0.28))

	# road trapezoid, projected from the player to the horizon
	var far := _project_z(VIEW_Z)
	var far_y: float = far.y
	var far_half: float = ROAD_HALF_PX * float(far.scale)
	var road := PackedVector2Array([
		Vector2(center_x - ROAD_HALF_PX - 14.0, view.y),
		Vector2(center_x + ROAD_HALF_PX + 14.0, view.y),
		Vector2(center_x + far_half + 10.0 * far.scale, far_y),
		Vector2(center_x - far_half - 10.0 * far.scale, far_y),
	])
	draw_colored_polygon(road, Color(0.14, 0.16, 0.19))
	var asphalt := PackedVector2Array([
		Vector2(center_x - ROAD_HALF_PX, view.y),
		Vector2(center_x + ROAD_HALF_PX, view.y),
		Vector2(center_x + far_half, far_y),
		Vector2(center_x - far_half, far_y),
	])
	draw_colored_polygon(asphalt, Color(0.075, 0.085, 0.105))
	draw_line(Vector2(center_x - ROAD_HALF_PX, view.y), Vector2(center_x - far_half, far_y), Color(0.6, 0.68, 0.76, 0.8), 3.0)
	draw_line(Vector2(center_x + ROAD_HALF_PX, view.y), Vector2(center_x + far_half, far_y), Color(0.6, 0.68, 0.76, 0.8), 3.0)

	_draw_roadside(scroll_z, center_x, view, horizon)
	for boundary in [-LANE_HALF_PX, LANE_HALF_PX]:
		_draw_lane_dashes(boundary, scroll_z, center_x)

	# draw far traffic first so near traffic and the player stay on top
	var traffic_list: Array = sim.traffic if state != "menu" else _menu_traffic(scroll_z)
	traffic_list = traffic_list.duplicate()
	traffic_list.sort_custom(func(a, b): return float(a.get("z", 0.0)) > float(b.get("z", 0.0)))
	for v in traffic_list:
		_draw_vehicle(v, center_x, scroll_z)

	if state != "menu":
		_draw_bike(player_y)

	# speed streaks outside the road edges
	var speed_ratio := clampf((sim.player_speed - sim.base_speed) / maxf(1.0, sim.max_speed - sim.base_speed), 0.0, 1.0)
	if speed_ratio > 0.05:
		for i in range(6):
			var sx := center_x - ROAD_HALF_PX - 26.0 - (i % 3) * 14.0
			var sy := fmod(scroll_z * PX_PER_M_NEAR * 1.4 + i * 97.0, view.y)
			draw_line(Vector2(sx, sy), Vector2(sx, sy + 30.0 + speed_ratio * 44.0), Color(0.55, 0.85, 1.0, 0.10 + speed_ratio * 0.22), 2.0)
			draw_line(Vector2(center_x + ROAD_HALF_PX + 26.0 + (i % 3) * 14.0, sy), Vector2(center_x + ROAD_HALF_PX + 26.0 + (i % 3) * 14.0, sy + 30.0 + speed_ratio * 44.0), Color(0.55, 0.85, 1.0, 0.10 + speed_ratio * 0.22), 2.0)

	if crash_flash > 0.0:
		draw_rect(Rect2(Vector2.ZERO, view), Color(1.0, 0.04, 0.02, clampf(crash_flash, 0.0, 0.55)))

	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)

func _draw_lane_dashes(boundary_px: float, scroll_z: float, center_x: float) -> void:
	var period := DASH_LENGTH_M + DASH_GAP_M
	var offset := fmod(scroll_z, period)
	var z := -offset
	while z < VIEW_Z:
		var z1 := maxf(z, 0.0)
		var z2 := minf(z + DASH_LENGTH_M, VIEW_Z)
		if z2 > z1:
			var p1 := _project_z(z1)
			var p2 := _project_z(z2)
			var x1: float = center_x + boundary_px * float(p1.scale)
			var x2: float = center_x + boundary_px * float(p2.scale)
			draw_line(Vector2(x1, p1.y), Vector2(x2, p2.y), Color(0.74, 0.82, 0.9, 0.75), maxf(1.0, 4.0 * p1.scale))
		z += period

func _draw_roadside(scroll_z: float, center_x: float, view: Vector2, horizon: float) -> void:
	# distant city silhouette
	for i in range(-4, 5):
		var bx := center_x + i * 84.0
		var bh := 34.0 + fmod(float(abs(i) * 31), 54.0)
		draw_rect(Rect2(bx - 30, horizon - bh, 60, bh), Color(0.01, 0.022, 0.038, 0.9))
		for wx in range(3):
			draw_rect(Rect2(bx - 18 + wx * 14, horizon - bh + 10, 5, 7), Color(0.35, 0.6, 0.72, 0.12))
	var post_period := 24.0
	var offset := fmod(scroll_z, post_period)
	var z := -offset
	while z < VIEW_Z:
		if z > 0.0:
			var p := _project_z(z)
			var half: float = (ROAD_HALF_PX + 26.0) * float(p.scale)
			var h: float = 22.0 * float(p.scale)
			for side in [-1, 1]:
				var px: float = center_x + float(side) * half
				draw_rect(Rect2(px - 2.0 * p.scale, p.y - h, 4.0 * p.scale, h), Color(0.3, 0.36, 0.42, 0.85))
				draw_rect(Rect2(px - 3.0 * p.scale, p.y - h - 2.0 * p.scale, 6.0 * p.scale, 3.0 * p.scale), Color(0.78, 0.86, 0.94, 0.85))
		z += post_period

func _menu_traffic(scroll_z: float) -> Array:
	var cars := []
	var specs := [
		{"lane": 0, "type": "car", "length": 4.5, "width": 1.8, "offset": 18.0},
		{"lane": 2, "type": "truck", "length": 7.0, "width": 2.4, "offset": 46.0},
		{"lane": 1, "type": "car", "length": 4.5, "width": 1.8, "offset": 74.0},
	]
	for i in range(specs.size()):
		var spec: Dictionary = specs[i]
		var z_rel := 8.0 + posmod(float(spec["offset"]) - menu_scroll * 0.5, 90.0)
		cars.append({
			"id": "menu_%d" % i,
			"lane": spec["lane"],
			"z": scroll_z + z_rel,
			"type": spec["type"],
			"length": spec["length"],
			"width": spec["width"],
		})
	return cars

func _draw_vehicle(v: Dictionary, center_x: float, scroll_z: float) -> void:
	var z_rel := float(v.get("z", 0.0)) - scroll_z
	if z_rel < -12.0 or z_rel > VIEW_Z:
		return
	var p := _project_z(z_rel)
	var scale: float = p.scale
	var lane := int(v.get("lane", 1))
	var vx := center_x - (sim.lane_center(lane) / 4.0) * LANE_PX * scale
	var dy: float = p.y
	var is_truck := String(v.get("type", "car")) == "truck"
	var length_px := float(v.get("length", 4.5)) * PX_PER_M_NEAR * scale
	var width_px := float(v.get("width", 1.8)) * PX_PER_M_NEAR * scale
	var palette := [
		Color(0.78, 0.2, 0.18),
		Color(0.18, 0.38, 0.78),
		Color(0.72, 0.74, 0.78),
		Color(0.2, 0.55, 0.36),
	]
	var color: Color = palette[abs(int(String(v.get("id", "0")).hash())) % palette.size()]
	if is_truck:
		color = Color(0.82, 0.52, 0.14)
	var body := Rect2(vx - width_px * 0.5, dy - length_px * 0.5, width_px, length_px)
	draw_rect(Rect2(body.position + Vector2(3, 5) * scale, body.size), Color(0, 0, 0, 0.34))
	var wheel_w := maxf(2.0, width_px * 0.16)
	var wheel_h := maxf(3.0, length_px * 0.18)
	for wx in [body.position.x - wheel_w * 0.45, body.end.x - wheel_w * 0.55]:
		for wy in [body.position.y + length_px * 0.14, body.position.y + length_px * 0.68]:
			draw_rect(Rect2(wx, wy, wheel_w, wheel_h), Color(0.02, 0.02, 0.03))
	draw_rect(body, color)
	draw_rect(body.grow(-2.0 * scale), color.lightened(0.06))
	var glass := Color(0.035, 0.07, 0.1)
	draw_rect(Rect2(body.position.x + width_px * 0.16, body.position.y + length_px * 0.18, width_px * 0.68, length_px * 0.24), glass)
	draw_rect(Rect2(body.position.x + width_px * 0.16, body.position.y + length_px * 0.58, width_px * 0.68, length_px * 0.16), glass.lightened(0.08))
	draw_rect(Rect2(body.position.x + width_px * 0.08, body.end.y - 3.0 * scale, width_px * 0.22, 3.0 * scale), Color(1.0, 0.12, 0.08, 0.95))
	draw_rect(Rect2(body.end.x - width_px * 0.30, body.end.y - 3.0 * scale, width_px * 0.22, 3.0 * scale), Color(1.0, 0.12, 0.08, 0.95))
	draw_rect(Rect2(body.position.x + width_px * 0.08, body.position.y + 1.0 * scale, width_px * 0.22, 2.0 * scale), Color(1.0, 0.94, 0.7, 0.45))
	draw_rect(Rect2(body.end.x - width_px * 0.30, body.position.y + 1.0 * scale, width_px * 0.22, 2.0 * scale), Color(1.0, 0.94, 0.7, 0.45))

func _draw_bike(player_y: float) -> void:
	var bx := _world_to_screen_x(sim.player_x)
	var lean := clampf(-sim.player_velocity_x * 0.02, -0.25, 0.25)
	var w := BIKE_WIDTH_M * PX_PER_M_NEAR
	var l := BIKE_LENGTH_M * PX_PER_M_NEAR
	# shadow, drawn without lean
	draw_set_transform(Vector2(bx + 4, player_y + 7) + shake_offset, 0.0, Vector2.ONE)
	draw_rect(Rect2(-w * 0.6, -l * 0.5, w * 1.2, l), Color(0, 0, 0, 0.35))
	draw_set_transform(Vector2(bx, player_y) + shake_offset, lean, Vector2.ONE)
	# wheels
	draw_rect(Rect2(-w * 0.18, -l * 0.50, w * 0.36, l * 0.22), Color(0.03, 0.03, 0.04))
	draw_rect(Rect2(-w * 0.18, l * 0.28, w * 0.36, l * 0.22), Color(0.03, 0.03, 0.04))
	draw_rect(Rect2(-w * 0.18, -l * 0.48, w * 0.36, l * 0.035), Color(0.38, 0.42, 0.48))
	draw_rect(Rect2(-w * 0.18, l * 0.44, w * 0.36, l * 0.035), Color(0.38, 0.42, 0.48))
	# frame and tank
	draw_rect(Rect2(-w * 0.28, -l * 0.28, w * 0.56, l * 0.58), Color(0.08, 0.42, 0.58))
	draw_rect(Rect2(-w * 0.30, -l * 0.14, w * 0.60, l * 0.26), Color(0.82, 0.22, 0.16))
	draw_rect(Rect2(-w * 0.24, -l * 0.10, w * 0.48, l * 0.08), Color(1.0, 0.5, 0.32, 0.55))
	# seat
	draw_rect(Rect2(-w * 0.24, l * 0.12, w * 0.48, l * 0.16), Color(0.09, 0.11, 0.14))
	# handlebars and arms
	draw_rect(Rect2(-w * 0.75, -l * 0.38, w * 1.5, l * 0.06), Color(0.62, 0.68, 0.74))
	draw_line(Vector2(-w * 0.36, -l * 0.08), Vector2(-w * 0.62, -l * 0.34), Color(0.12, 0.15, 0.2), maxf(2.0, w * 0.14))
	draw_line(Vector2(w * 0.36, -l * 0.08), Vector2(w * 0.62, -l * 0.34), Color(0.12, 0.15, 0.2), maxf(2.0, w * 0.14))
	# rider torso, shoulders and helmet
	draw_colored_polygon(PackedVector2Array([
		Vector2(-w * 0.34, -l * 0.10), Vector2(w * 0.34, -l * 0.10),
		Vector2(w * 0.28, l * 0.16), Vector2(-w * 0.28, l * 0.16)
	]), Color(0.10, 0.13, 0.18))
	draw_circle(Vector2(0, -l * 0.15), w * 0.34, Color(0.08, 0.10, 0.14))
	draw_circle(Vector2(0, -l * 0.20), w * 0.27, Color(0.9, 0.93, 0.96))
	draw_rect(Rect2(-w * 0.22, -l * 0.23, w * 0.44, l * 0.035), Color(0.12, 0.65, 0.82))
	# taillight and headlight
	draw_rect(Rect2(-w * 0.12, l * 0.42, w * 0.24, l * 0.035), Color(1.0, 0.1, 0.06, 0.95))
	draw_rect(Rect2(-w * 0.12, -l * 0.50, w * 0.24, l * 0.035), Color(1.0, 0.94, 0.68))
	if sim.braking:
		draw_circle(Vector2(0, l * 0.45), w * 0.22, Color(1.0, 0.06, 0.03, 0.75))
	if external_control:
		draw_arc(Vector2.ZERO, l * 0.62, 0.0, TAU, 32, Color(0.2, 0.9, 1.0, 0.45), 2.0)
	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

func _build_ui() -> void:
	ui_layer = CanvasLayer.new()
	ui_layer.process_mode = Node.PROCESS_MODE_ALWAYS
	add_child(ui_layer)
	var ui_theme := _make_theme()

	hud = Control.new()
	hud.set_anchors_preset(Control.PRESET_FULL_RECT)
	hud.theme = ui_theme
	ui_layer.add_child(hud)

	menu_backdrop = ColorRect.new()
	menu_backdrop.set_anchors_preset(Control.PRESET_FULL_RECT)
	menu_backdrop.color = Color(0.0, 0.0, 0.0, 0.62)
	menu_backdrop.mouse_filter = Control.MOUSE_FILTER_IGNORE
	menu_backdrop.visible = false
	ui_layer.add_child(menu_backdrop)

	var score_panel := _make_hud_panel(Vector2(0, 0), Vector2(24, 20), Vector2(230, 104))
	hud.add_child(score_panel)
	var score_box := VBoxContainer.new()
	score_box.add_theme_constant_override("separation", 1)
	score_panel.add_child(score_box)
	score_label = Label.new()
	score_label.text = "0"
	score_label.add_theme_font_size_override("font_size", 32)
	score_label.add_theme_color_override("font_color", Color(0.94, 0.98, 1.0))
	score_box.add_child(score_label)
	best_label = Label.new()
	best_label.text = "BEST 0"
	best_label.add_theme_font_size_override("font_size", 12)
	best_label.add_theme_color_override("font_color", Color(0.55, 0.66, 0.72))
	score_box.add_child(best_label)
	dist_label = Label.new()
	dist_label.text = "0 m"
	dist_label.add_theme_font_size_override("font_size", 12)
	dist_label.add_theme_color_override("font_color", Color(0.55, 0.66, 0.72))
	score_box.add_child(dist_label)

	var right_panel := _make_hud_panel(Vector2(1, 0), Vector2(-24, 20), Vector2(270, 104))
	hud.add_child(right_panel)
	var right_box := VBoxContainer.new()
	right_box.add_theme_constant_override("separation", 1)
	right_panel.add_child(right_box)
	speed_label = Label.new()
	speed_label.text = "0 km/h"
	speed_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	speed_label.add_theme_font_size_override("font_size", 30)
	right_box.add_child(speed_label)
	status_label = Label.new()
	status_label.text = "HUMAN"
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	status_label.add_theme_font_size_override("font_size", 12)
	status_label.add_theme_color_override("font_color", Color(0.55, 0.66, 0.72))
	right_box.add_child(status_label)
	var record_row := HBoxContainer.new()
	record_row.alignment = BoxContainer.ALIGNMENT_END
	record_row.add_theme_constant_override("separation", 8)
	right_box.add_child(record_row)
	record_button = Button.new()
	record_button.text = "Record"
	record_button.custom_minimum_size = Vector2(88, 30)
	record_button.focus_mode = Control.FOCUS_NONE
	record_button.pressed.connect(_on_record_button_pressed)
	record_row.add_child(record_button)
	record_status_label = Label.new()
	record_status_label.text = ""
	record_status_label.custom_minimum_size = Vector2(120, 0)
	record_status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	record_status_label.add_theme_font_size_override("font_size", 11)
	record_row.add_child(record_status_label)

	var controls_panel := _make_hud_panel(Vector2(0, 1), Vector2(24, -24), Vector2(470, 40))
	hud.add_child(controls_panel)
	msg_label = Label.new()
	msg_label.text = "A / D  steer     SPACE  brake     T  hand control     ESC  pause     F9  record     F11  fullscreen"
	msg_label.add_theme_font_size_override("font_size", 12)
	msg_label.add_theme_color_override("font_color", Color(0.62, 0.72, 0.78))
	controls_panel.add_child(msg_label)

	ai_panel = _make_hud_panel(Vector2(1, 1), Vector2(-24, -24), Vector2(330, 92))
	hud.add_child(ai_panel)
	var ai_box := VBoxContainer.new()
	ai_box.add_theme_constant_override("separation", 1)
	ai_panel.add_child(ai_box)
	ai_model_label = Label.new()
	ai_model_label.text = "AI  HUMAN CONTROL"
	ai_model_label.add_theme_font_size_override("font_size", 13)
	ai_model_label.add_theme_color_override("font_color", Color(0.35, 0.9, 0.95))
	ai_box.add_child(ai_model_label)
	ai_action_label = Label.new()
	ai_action_label.text = "ACTION  --"
	ai_action_label.add_theme_font_size_override("font_size", 16)
	ai_box.add_child(ai_action_label)
	ai_latency_label = Label.new()
	ai_latency_label.text = "LATENCY  --"
	ai_latency_label.add_theme_font_size_override("font_size", 11)
	ai_latency_label.add_theme_color_override("font_color", Color(0.55, 0.66, 0.72))
	ai_box.add_child(ai_latency_label)
	conn_label = Label.new()
	conn_label.text = "WS  disconnected"
	conn_label.add_theme_font_size_override("font_size", 10)
	conn_label.add_theme_color_override("font_color", Color(0.45, 0.55, 0.62))
	ai_box.add_child(conn_label)

	notice_label = Label.new()
	notice_label.set_anchors_preset(Control.PRESET_TOP_WIDE)
	notice_label.offset_top = 132
	notice_label.offset_bottom = 162
	notice_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	notice_label.add_theme_color_override("font_color", Color(0.4, 0.95, 0.85))
	notice_label.add_theme_font_size_override("font_size", 20)
	notice_label.visible = false
	hud.add_child(notice_label)

	countdown_label = Label.new()
	countdown_label.set_anchors_preset(Control.PRESET_FULL_RECT)
	countdown_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	countdown_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	countdown_label.add_theme_font_size_override("font_size", 96)
	countdown_label.add_theme_color_override("font_color", Color(0.3, 0.95, 0.95))
	countdown_label.visible = false
	hud.add_child(countdown_label)

	menu_panel = _make_panel("LANESHIFT", "THREE LANES. ONE MODEL.", ["Start", "Watch Kev", "Open Logs", "Quit"])
	menu_panel.theme = ui_theme
	ui_layer.add_child(menu_panel)
	pause_panel = _make_panel("PAUSED", "", ["Resume", "Restart", "Menu"])
	pause_panel.theme = ui_theme
	pause_panel.visible = false
	ui_layer.add_child(pause_panel)
	gameover_panel = _make_panel("GAME OVER", "", ["Restart", "Menu"])
	gameover_panel.theme = ui_theme
	gameover_panel.visible = false
	ui_layer.add_child(gameover_panel)
	gameover_result_label = gameover_panel.find_child("Result", true, false) as Label
	if gameover_result_label == null:
		gameover_result_label = Label.new()
		gameover_result_label.name = "Result"
		gameover_result_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		(gameover_panel.get_child(0) as VBoxContainer).add_child(gameover_result_label)

	_set_state("menu")
	_wire_menu()

func _make_theme() -> Theme:
	var theme := Theme.new()
	theme.set_color("font_color", "Label", Color(0.88, 0.94, 0.98))
	theme.set_color("font_color", "Button", Color(0.9, 0.97, 1.0))
	theme.set_font_size("font_size", "Label", 15)
	theme.set_font_size("font_size", "Button", 17)
	theme.set_constant("separation", "HBoxContainer", 8)
	theme.set_constant("separation", "VBoxContainer", 8)
	var panel_style := StyleBoxFlat.new()
	panel_style.bg_color = Color(0.02, 0.05, 0.08, 0.88)
	panel_style.border_color = Color(0.1, 0.55, 0.62, 0.5)
	panel_style.set_border_width_all(1)
	panel_style.set_corner_radius_all(10)
	panel_style.content_margin_left = 14
	panel_style.content_margin_right = 14
	panel_style.content_margin_top = 10
	panel_style.content_margin_bottom = 10
	theme.set_stylebox("panel", "PanelContainer", panel_style)
	var button_style := StyleBoxFlat.new()
	button_style.bg_color = Color(0.04, 0.13, 0.17)
	button_style.set_corner_radius_all(6)
	button_style.content_margin_left = 12
	button_style.content_margin_right = 12
	button_style.content_margin_top = 7
	button_style.content_margin_bottom = 7
	var button_hover := button_style.duplicate()
	button_hover.bg_color = Color(0.06, 0.3, 0.35)
	var button_pressed := button_style.duplicate()
	button_pressed.bg_color = Color(0.04, 0.5, 0.55)
	theme.set_stylebox("normal", "Button", button_style)
	theme.set_stylebox("hover", "Button", button_hover)
	theme.set_stylebox("pressed", "Button", button_pressed)
	var focus_style := button_style.duplicate()
	focus_style.bg_color = Color(0.08, 0.4, 0.46)
	theme.set_stylebox("focus", "Button", focus_style)
	return theme

func _make_hud_panel(anchor: Vector2, offset: Vector2, panel_size: Vector2) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.anchor_left = anchor.x
	panel.anchor_right = anchor.x
	panel.anchor_top = anchor.y
	panel.anchor_bottom = anchor.y
	if anchor.x > 0.5:
		panel.offset_left = offset.x - panel_size.x
		panel.offset_right = offset.x
		panel.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	else:
		panel.offset_left = offset.x
		panel.offset_right = offset.x + panel_size.x
	if anchor.y > 0.5:
		panel.offset_top = offset.y - panel_size.y
		panel.offset_bottom = offset.y
		panel.grow_vertical = Control.GROW_DIRECTION_BEGIN
	else:
		panel.offset_top = offset.y
		panel.offset_bottom = offset.y + panel_size.y
	panel.custom_minimum_size = panel_size
	return panel

func _make_panel(title: String, tagline: String, buttons: Array) -> Control:
	var panel := PanelContainer.new()
	var panel_size := Vector2(420, 200.0 + buttons.size() * 48.0)
	panel.anchor_left = 0.5
	panel.anchor_right = 0.5
	panel.anchor_top = 0.5
	panel.anchor_bottom = 0.5
	panel.offset_left = -panel_size.x * 0.5
	panel.offset_right = panel_size.x * 0.5
	panel.offset_top = -panel_size.y * 0.5
	panel.offset_bottom = panel_size.y * 0.5
	panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	panel.grow_vertical = Control.GROW_DIRECTION_BOTH
	panel.custom_minimum_size = panel_size
	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 10)
	panel.add_child(vbox)
	var t := Label.new()
	t.text = title
	t.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	t.add_theme_color_override("font_color", Color(0.35, 0.91, 0.94))
	t.add_theme_font_size_override("font_size", 36)
	vbox.add_child(t)
	if tagline != "":
		var tag := Label.new()
		tag.text = tagline
		tag.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		tag.add_theme_color_override("font_color", Color(0.55, 0.68, 0.74))
		tag.add_theme_font_size_override("font_size", 12)
		vbox.add_child(tag)
	if title == "GAME OVER":
		gameover_result_label = Label.new()
		gameover_result_label.name = "Result"
		gameover_result_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		gameover_result_label.add_theme_color_override("font_color", Color(1.0, 0.55, 0.45))
		gameover_result_label.add_theme_font_size_override("font_size", 14)
		gameover_result_label.text = ""
		vbox.add_child(gameover_result_label)
	for bname in buttons:
		var btn := Button.new()
		btn.text = bname
		btn.name = bname
		btn.custom_minimum_size = Vector2(260, 44)
		btn.focus_mode = Control.FOCUS_ALL
		vbox.add_child(btn)
	return panel

func _wire_menu() -> void:
	for btn in _find_buttons(menu_panel):
		btn.pressed.connect(func(): _on_menu_button(btn.text))
	for btn in _find_buttons(pause_panel):
		btn.pressed.connect(func(): _on_pause_button(btn.text))
	for btn in _find_buttons(gameover_panel):
		btn.pressed.connect(func(): _on_gameover_button(btn.text))

func _set_state(next: String) -> void:
	state = next
	menu_panel.visible = next == "menu"
	pause_panel.visible = next == "paused"
	gameover_panel.visible = next == "gameover" and crash_hold <= 0.0
	hud.visible = next == "playing" or next == "countdown"
	menu_backdrop.visible = next == "menu" or next == "paused" or next == "gameover"
	countdown_label.visible = next == "countdown"
	get_tree().paused = next == "paused"
	if next == "playing" or next == "countdown":
		get_viewport().gui_release_focus()
	elif next == "menu":
		_focus_button(menu_panel, "Start")
	elif next == "paused":
		_focus_button(pause_panel, "Resume")
	elif next == "gameover" and gameover_panel.visible:
		_focus_button(gameover_panel, "Restart")

func _focus_button(panel: Control, button_name: String) -> void:
	var target := panel.find_child(button_name, true, false)
	if target is Button:
		target.grab_focus()

func _find_buttons(root: Node) -> Array:
	var res := []
	for c in root.get_children():
		if c is Button:
			res.append(c)
		res += _find_buttons(c)
	return res

func _on_menu_button(text: String) -> void:
	_play_sfx(_make_wav(_noise_samples(0.035, 2.0, 880.0), false))
	if text == "Start":
		external_control = false
		control_mode = "human"
		start_game()
	elif text == "Watch Kev":
		if net != null and net.connected_flag:
			external_control = true
			control_mode = "external"
			start_game()
		else:
			_show_notice("Start the local Kev controller first: PLAY-KEV.bat")
	elif text == "Open Logs":
		var runs_path := PortablePathsScript.get_recordings_root()
		DirAccess.make_dir_recursive_absolute(runs_path)
		OS.shell_open(runs_path)
	elif text == "Quit":
		if sim.status == "running":
			sim.finish_episode("quit")
		if telemetry != null:
			telemetry.close()
		get_tree().quit()

func _on_pause_button(text: String) -> void:
	_play_sfx(_make_wav(_noise_samples(0.035, 2.0, 880.0), false))
	if text == "Resume":
		resume_game()
	elif text == "Restart":
		restart_game()
	elif text == "Menu":
		go_menu()

func _on_gameover_button(text: String) -> void:
	_play_sfx(_make_wav(_noise_samples(0.035, 2.0, 880.0), false))
	if text == "Restart":
		restart_game()
	elif text == "Menu":
		go_menu()

func start_game(seed: int = -1, cfg: Dictionary = {}, requested_episode_id: String = "") -> void:
	var previous_episode_id := sim.episode_id
	var is_reset := previous_episode_id != "" and sim.status != "idle"
	if sim.status == "running":
		sim.finish_episode("reset")
	if seed < 0:
		seed = rng_visual.randi()
	var active_config := DEFAULT_CONFIG.duplicate(true)
	for key in cfg:
		active_config[key] = cfg[key]
	sim.reset(seed, active_config, requested_episode_id)
	action_event_sequence = 0
	last_score_milestone = 0
	decision_state_history.clear()
	last_external_action_msec = Time.get_ticks_msec()
	set_meta("prev_left", false)
	set_meta("prev_right", false)
	get_tree().paused = false
	if auto_run:
		control_mode = "auto"
		external_control = true
	elif not external_control or not net.connected_flag:
		control_mode = "human"
		external_control = false
	else:
		control_mode = "external"
	sim.control_source = "external_ai" if external_control else "human"
	net.set_episode_id(sim.episode_id)
	if is_reset:
		_log_event("episode_reset", {"previous_episode_id": previous_episode_id, "new_episode_id": sim.episode_id, "scenario_seed": seed})
	_log_event("episode_start", {"scenario_seed": seed, "config": active_config})
	crash_hold = 0.0
	crash_flash = 0.0
	if countdown_seconds > 0.0:
		countdown_timer = countdown_seconds
		countdown_label.text = str(int(ceil(countdown_timer)))
		_set_state("countdown")
	else:
		_set_state("playing")
	_update_ai_status()
	telemetry_timer = 0.0
	decision_timer = 0.0
	queue_redraw()

func restart_game() -> void:
	start_game(rng_visual.randi())

func go_menu() -> void:
	if sim.status == "running":
		sim.finish_episode("menu_exit")
	# A menu is a presentation-only state. Do not reset the simulation or emit
	# scenario events for it; the next start_game creates a fresh episode.
	sim.traffic.clear()
	sim.status = "idle"
	sim.player_speed = 0.0
	sim.player_x = 0.0
	sim.player_z = 0.0
	_set_state("menu")
	_update_ai_status()
	_update_hud()
	queue_redraw()

func _on_crashed(_eid: String) -> void:
	shake_timer = 0.6
	shake_intensity = 0.6
	crash_flash = 0.55
	crash_hold = 0.7
	_play_sfx(_make_wav(_noise_samples(0.35, 2.2, 70.0), false))
	_log_event("collision", {"collision_count": sim.collision_count})

func _on_vehicle_overtaken(vehicle_id: String) -> void:
	notice_label.text = "CLEAN PASS  +25"
	notice_label.visible = true
	notice_timer = 1.1
	_log_event("vehicle_overtaken", {"vehicle_id": vehicle_id, "score": sim.score})

func _on_episode_ended(summary: Dictionary) -> void:
	var terminal_state: Dictionary = sim.get_snapshot()
	terminal_state["state_id"] = "%s:%d:terminal" % [sim.episode_id, sim.physics_tick]
	terminal_state["snapshot_kind"] = "terminal"
	terminal_state["wall_time_unix"] = Time.get_unix_time_from_system()
	terminal_state["wall_time_monotonic_msec"] = Time.get_ticks_msec()
	telemetry.log_telemetry(terminal_state)
	var end_event := summary.duplicate(true)
	end_event["terminal_state_id"] = terminal_state["state_id"]
	_log_event("episode_end", end_event)
	telemetry.log_episode(summary)
	print("LANESHIFT_EPISODE " + JSON.stringify(summary))
	if summary.get("score", 0.0) > best_score:
		best_score = summary["score"]
		_save_highscore()
	if auto_running:
		call_deferred("_auto_next")
	else:
		if String(summary.get("termination_reason", "")) != "crash":
			crash_hold = 0.0
		gameover_result_label.text = "Score %.0f   Distance %.0f m\nSurvived %.1f s" % [summary["score"], summary["distance"], summary["survival_time"]]
		_set_state("gameover")
		_update_ai_status()

func _on_scenario_init(info: Dictionary) -> void:
	if telemetry != null:
		_log_event("scenario_initialized", info)

func _on_net_action(action: String, duration: float, seq: int, tick: int, episode_id: String, label_source: String, decision_metadata: Dictionary) -> void:
	var observed_state: Dictionary = _get_decision_state(tick)
	_log_event("external_action_received", {
		"controller_seq": seq,
		"command_episode_id": episode_id,
		"command_tick": tick,
		"decision_state_id": observed_state.get("state_id", ""),
		"requested_action": action,
		"requested_duration": duration,
		"label_source": label_source,
		"decision_metadata": decision_metadata
	})
	_update_ai_panel(action, label_source, decision_metadata)
	if state != "playing":
		_log_event("external_action_ignored", {"controller_seq": seq, "requested_action": action, "reason": "episode_not_playing", "decision_metadata": decision_metadata})
		return
	if (external_control or auto_run) and episode_id == sim.episode_id:
		var decision_state: Dictionary = observed_state
		if decision_state.is_empty():
			_log_event("external_action_rejected", {
				"controller_seq": seq,
				"requested_action": action,
				"command_tick": tick,
				"decision_metadata": decision_metadata,
				"reason": "decision_state_unavailable"
			})
			return
		if action not in decision_state.get("legal_actions", []):
			_log_event("external_action_rejected", {
				"controller_seq": seq,
				"requested_action": action,
				"command_tick": tick,
				"decision_state_id": decision_state.get("state_id", ""),
				"label_source": label_source,
				"decision_metadata": decision_metadata,
				"reason": "action_not_legal_in_decision_state"
			})
			return
		var actual_source := label_source if label_source in ["rule_based", "original_model", "fine_tuned_model", "controller_fallback"] else "external_ai"
		last_external_action_msec = Time.get_ticks_msec()
		var applied: bool = _submit_action(action, duration, actual_source, label_source, seq, tick, decision_state, decision_metadata)
		_log_event("external_action_applied" if applied else "external_action_rejected", {
			"controller_seq": seq,
			"requested_action": action,
			"requested_duration": duration,
			"applied_action": action if applied else sim.current_action,
			"accepted": applied,
			"decision_metadata": decision_metadata,
			"reason": "" if applied else "not_legal_at_application"
		})
	else:
		_log_event("external_action_ignored", {"controller_seq": seq, "requested_action": action, "reason": "human_control_active", "decision_metadata": decision_metadata})

func _on_command_rejected(seq: int, reason: String, episode_id: String, tick: int, decision_metadata: Dictionary) -> void:
	_log_event("external_action_rejected", {"controller_seq": seq, "reason": reason, "command_episode_id": episode_id, "command_tick": tick, "decision_metadata": decision_metadata})

func _on_net_reset(seed: int, cfg: Dictionary, episode_id: String) -> void:
	external_control = true
	control_mode = "external"
	start_game(seed, cfg, episode_id)
	_update_ai_status()

func _on_net_connected() -> void:
	conn_label.text = "WS  connected"
	_log_event("controller_connected", {})
	_update_ai_status()

func _on_net_disconnected() -> void:
	conn_label.text = "WS  disconnected"
	if not auto_run:
		external_control = false
		control_mode = "human"
	_log_event("controller_disconnected", {})
	_update_ai_status()

func _submit_action(action: String, duration: float, control_source: String, label_source: String, controller_seq: int = -1, state_tick: int = -1, decision_state: Dictionary = {}, decision_metadata: Dictionary = {}, count_action: bool = true) -> bool:
	var application_state: Dictionary = sim.get_snapshot()
	var state_before: Dictionary = decision_state.duplicate(true) if not decision_state.is_empty() else application_state
	var decision_tick := int(state_before.get("physics_tick", state_tick if state_tick >= 0 else sim.physics_tick))
	var application_tick := sim.physics_tick
	var accepted: bool = sim.apply_action(action, duration, control_source, count_action)
	if accepted and action in ["LEFT", "RIGHT"]:
		_play_sfx(_make_wav(_noise_samples(0.12, 1.6, 520.0), false))
	action_event_sequence += 1
	var applied_action := action if accepted else sim.current_action
	_log_event("action_decision", {
		"event_id": "%s:action:%d" % [sim.episode_id, action_event_sequence],
		"decision_seq": action_event_sequence,
		"controller_seq": controller_seq,
		"state_id": state_before.get("state_id", ""),
		"physics_tick": decision_tick,
		"decision_tick": decision_tick,
		"application_tick": application_tick,
		"application_state_id": application_state.get("state_id", ""),
		"application_latency_ticks": max(0, application_tick - decision_tick),
		"decision_sim_time": state_before.get("sim_time", sim.sim_time),
		"application_sim_time": application_state.get("sim_time", sim.sim_time),
		"controller_state_tick": state_tick if state_tick >= 0 else decision_tick,
		"requested_action": action,
		"requested_duration": duration,
		"applied_action": applied_action,
		"applied_duration": sim.action_timer if accepted else 0.0,
		"accepted": accepted,
		"control_source": control_source,
		"label_source": label_source,
		"decision_metadata": decision_metadata,
		"state_before": state_before
	})
	return accepted

func _log_score_milestones(current_score: float) -> void:
	var reached_milestone := int(floor(current_score / 100.0)) * 100
	while reached_milestone > last_score_milestone:
		last_score_milestone += 100
		_log_event("score_milestone", {"milestone": last_score_milestone, "score": current_score})

func _remember_decision_state(snapshot: Dictionary) -> void:
	var tick := int(snapshot.get("physics_tick", -1))
	if tick < 0 or snapshot.get("episode_id", "") != sim.episode_id:
		return
	decision_state_history[tick] = snapshot.duplicate(true)
	var oldest_kept := tick - int(decision_state_history_seconds * 60.0)
	for stored_tick in decision_state_history.keys():
		if int(stored_tick) < oldest_kept:
			decision_state_history.erase(stored_tick)

func _get_decision_state(tick: int) -> Dictionary:
	var snapshot: Dictionary = decision_state_history.get(tick, {})
	if snapshot.get("episode_id", "") != sim.episode_id:
		return {}
	return snapshot.duplicate(true)

func _log_event(event_type: String, fields: Dictionary) -> void:
	if telemetry == null:
		return
	var event := {
		"type": event_type,
		"episode_id": sim.episode_id if sim != null else "",
		"sim_time": sim.sim_time if sim != null else 0.0,
		"physics_tick": sim.physics_tick if sim != null else 0,
		"wall_time_unix": Time.get_unix_time_from_system(),
		"wall_time_monotonic_msec": Time.get_ticks_msec()
	}
	for key in fields:
		event[key] = fields[key]
	telemetry.log_event(event)
	if net != null:
		net.send_event(event)

func _physics_process(delta: float) -> void:
	if recording_active and telemetry != null:
		telemetry.tick(delta)
	if notice_timer > 0.0:
		notice_timer = max(0.0, notice_timer - delta)
		notice_label.visible = notice_timer > 0.0
	if shake_timer > 0:
		shake_timer -= delta
		if shake_timer <= 0:
			shake_offset = Vector2.ZERO
	if crash_flash > 0.0:
		crash_flash = maxf(0.0, crash_flash - delta * 0.9)
	if state == "gameover" and crash_hold > 0.0:
		crash_hold = maxf(0.0, crash_hold - delta)
		if crash_hold <= 0.0:
			gameover_panel.visible = true
			_focus_button(gameover_panel, "Restart")
	if state == "countdown":
		countdown_timer = maxf(0.0, countdown_timer - delta)
		if countdown_timer > 0.0:
			countdown_label.text = str(int(ceil(countdown_timer)))
		else:
			_set_state("playing")
			_log_event("countdown_finished", {})
		_update_hud()
		queue_redraw()
		return
	if state == "playing":
		_handle_human_input(delta)
		if external_control and not auto_run and Time.get_ticks_msec() - last_external_action_msec > external_timeout_msec:
			external_control = false
			control_mode = "human"
			_log_event("controller_timeout", {"fallback_control": "human"})
			_submit_action("STAY", 0.1, "human", "human")
		sim.step(delta * sim_speed_scale)
		_log_score_milestones(sim.score)
		if sim.status != "running":
			_update_hud()
			queue_redraw()
			return
		if recording_active:
			telemetry_timer += delta
			if telemetry_timer >= 1.0 / telemetry_hz:
				telemetry_timer = 0.0
				var snap: Dictionary = sim.get_snapshot()
				snap["wall_time_unix"] = Time.get_unix_time_from_system()
				snap["wall_time_monotonic_msec"] = Time.get_ticks_msec()
				if telemetry != null:
					telemetry.log_telemetry(snap)
		decision_timer += delta
		if decision_timer >= 1.0 / decision_hz:
			decision_timer = 0.0
			var snap2: Dictionary = sim.get_snapshot()
			snap2["wall_time_unix"] = Time.get_unix_time_from_system()
			snap2["wall_time_monotonic_msec"] = Time.get_ticks_msec()
			_remember_decision_state(snap2)
			net.update_state(snap2)
			net.send_state(snap2)
		_update_hud()
	queue_redraw()

func _handle_human_input(_delta: float) -> void:
	if external_control:
		return
	var brake_now := Input.is_key_pressed(KEY_SPACE) or Input.is_key_pressed(KEY_DOWN)
	if brake_now:
		if not brake_held:
			brake_held = true
			_submit_action("BRAKE", 0.15, "human", "human")
		else:
			sim.apply_action("BRAKE", 0.15, "human", false)
	else:
		if brake_held:
			brake_held = false
			_submit_action("STAY", 0.1, "human", "human")

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if state != "playing" or external_control:
			return
		match event.keycode:
			KEY_A, KEY_LEFT:
				_submit_action("LEFT", 0.2, "human", "human")
			KEY_D, KEY_RIGHT:
				_submit_action("RIGHT", 0.2, "human", "human")

func _process(_delta: float) -> void:
	if state == "menu":
		menu_scroll += _delta * 12.0
	if _key_just_pressed(KEY_F9):
		_toggle_recording()
	if _key_just_pressed(KEY_F11):
		var current := DisplayServer.window_get_mode()
		if current == DisplayServer.WINDOW_MODE_FULLSCREEN:
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
		else:
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
	if Input.is_action_just_pressed("ui_cancel"):
		if state == "playing":
			pause_game()
		elif state == "paused":
			resume_game()
	if state == "playing" and _key_just_pressed(KEY_T):
		if external_control:
			external_control = false
			control_mode = "human"
			_submit_action("STAY", 0.1, "human", "human")
		else:
			if net.connected_flag:
				external_control = true
				control_mode = "external"
				last_external_action_msec = Time.get_ticks_msec()
				sim.control_source = "external_ai"
				_log_event("control_mode_changed", {"control_mode": "external"})
			else:
				_log_event("control_mode_change_rejected", {"requested_control_mode": "external", "reason": "controller_disconnected"})
		_update_ai_status()
	conn_label.text = "WS  " + ("connected" if net.connected_flag else "disconnected")
	if state == "playing" and take_control_on_start and net.connected_flag and not external_control:
		external_control = true
		control_mode = "external"
		last_external_action_msec = Time.get_ticks_msec()
		sim.control_source = "external_ai"
		take_control_on_start = false
		_log_event("control_mode_changed", {"control_mode": "external", "reason": "take_control_on_start"})
		_update_ai_status()
	if state == "gameover" and _key_just_pressed(KEY_R):
		restart_game()
	_update_record_ui()
	if recording_active and record_status_label != null:
		var elapsed := float(Time.get_ticks_msec() - recording_start_msec) / 1000.0
		var samples := 0
		var events := 0
		if telemetry != null:
			samples = telemetry.telemetry_record_count
			events = telemetry.event_record_count
		record_status_label.text = "REC %.1fs  %d samples  %d events" % [elapsed, samples, events]
	if shake_timer > 0.0:
		shake_offset = Vector2(randf_range(-shake_intensity, shake_intensity), randf_range(-shake_intensity, shake_intensity)) * 4.0
	_update_audio()

func _key_just_pressed(key: int) -> bool:
	if not has_meta("keys_prev"):
		set_meta("keys_prev", {})
	var d: Dictionary = get_meta("keys_prev")
	var now := Input.is_key_pressed(key)
	var prev := bool(d.get(key, false))
	d[key] = now
	set_meta("keys_prev", d)
	return now and not prev

func pause_game() -> void:
	if state != "playing":
		return
	_log_event("pause", {})
	_set_state("paused")

func resume_game() -> void:
	if state != "paused":
		return
	_log_event("resume", {})
	_set_state("playing")

func _toggle_recording() -> void:
	if recording_active:
		stop_recording()
	else:
		start_recording()

func start_recording() -> void:
	if recording_active:
		return
	var root := PortablePathsScript.get_portable_root()
	if not PortablePathsScript.is_writable(root):
		_show_notice("Recording failed: portable root is not writable. Extract to a writable folder.")
		return
	PortablePathsScript.ensure_dir(PortablePathsScript.get_recordings_root())
	_read_build_info()
	if telemetry == null or telemetry.writer_closed or not telemetry.enabled or not telemetry.session_active:
		if telemetry != null:
			telemetry.queue_free()
		telemetry = TelemetryLogger.new()
		add_child(telemetry)
	telemetry.build_commit = build_commit
	telemetry.build_godot_version = build_godot_version
	telemetry.start_session(PortablePathsScript.get_recordings_root())
	if not telemetry.enabled:
		_show_notice("Recording failed: telemetry writer did not start.")
		return
	recording_active = true
	recording_session_id = telemetry.session_id
	recording_start_msec = Time.get_ticks_msec()
	_create_recording_lock()
	_log_event("recording_started", {"session_id": recording_session_id})
	_update_record_ui()

func stop_recording() -> void:
	if not recording_active:
		return
	_log_event("recording_stopped", {"session_id": recording_session_id})
	if telemetry != null:
		telemetry.close()
	recording_active = false
	_remove_recording_lock()
	var rel_path := ""
	if telemetry != null:
		rel_path = telemetry.session_dir
	_show_notice("Recording saved to " + rel_path)
	_update_record_ui()

func _create_recording_lock() -> void:
	PortablePathsScript.ensure_dir(PortablePathsScript.get_recordings_root())
	var f := FileAccess.open(PortablePathsScript.get_recordings_root().path_join(".recording_active"), FileAccess.WRITE)
	if f:
		f.store_string(recording_session_id)
		f.close()

func _remove_recording_lock() -> void:
	DirAccess.remove_absolute(PortablePathsScript.get_recordings_root().path_join(".recording_active"))

func _read_build_info() -> void:
	build_commit = "dev"
	build_godot_version = "%d.%d.%d" % [Engine.get_version_info()["major"], Engine.get_version_info()["minor"], Engine.get_version_info()["patch"]]
	var root := PortablePathsScript.get_portable_root()
	var candidates := [root.path_join("BUILD-INFO.txt"), "res://BUILD-INFO.txt"]
	for path in candidates:
		if FileAccess.file_exists(path):
			var f := FileAccess.open(path, FileAccess.READ)
			if f:
				while not f.eof_reached():
					var line := f.get_line().strip_edges()
					if line == "" or line.begins_with("#"):
						continue
					var parts := line.split("=", true, 1)
					if parts.size() == 2:
						var key := parts[0].strip_edges()
						var val := parts[1].strip_edges()
						if key == "commit" or key == "git_commit" or key == "commit_sha":
							build_commit = val
						elif key == "godot_version" or key == "engine_version":
							build_godot_version = val
				f.close()
			return

func _show_notice(text: String) -> void:
	notice_label.text = text
	notice_label.visible = true
	notice_timer = 3.0

func _on_record_button_pressed() -> void:
	_play_sfx(_make_wav(_noise_samples(0.035, 2.0, 880.0), false))
	_toggle_recording()

func _update_record_ui() -> void:
	if record_button == null:
		return
	if recording_active:
		record_button.text = "Stop"
		record_status_label.text = "REC"
		record_status_label.add_theme_color_override("font_color", Color(1.0, 0.35, 0.3))
	else:
		record_button.text = "Record"
		record_status_label.text = ""
		record_status_label.add_theme_color_override("font_color", Color(0.55, 0.66, 0.72))

func _update_hud() -> void:
	score_label.text = "%.0f" % sim.score
	best_label.text = "BEST %.0f" % best_score
	dist_label.text = "%.0f m" % sim.distance
	speed_label.text = "%.0f km/h" % (sim.player_speed * 3.6)
	status_label.text = "CONTROL  " + control_mode.to_upper()

func _update_ai_status() -> void:
	if ai_model_label == null:
		return
	if net == null or not net.connected_flag:
		ai_model_label.text = "AI  OFFLINE" if external_control else "AI  HUMAN CONTROL"
		ai_action_label.text = "ACTION  --"
		ai_latency_label.text = "LATENCY  --"
	elif external_control or auto_run:
		ai_model_label.text = "AI  " + ("AUTO" if auto_run else "EXTERNAL CONTROL")
		ai_action_label.text = "ACTION  waiting"
		ai_latency_label.text = "MODEL  loading"
	else:
		ai_model_label.text = "AI  READY"
		ai_action_label.text = "ACTION  --"
		ai_latency_label.text = "PRESS T TO HAND OVER"

func _build_audio() -> void:
	if DisplayServer.get_name() == "headless":
		return
	engine_player = AudioStreamPlayer.new()
	engine_player.stream = _make_wav(_engine_samples(), true)
	engine_player.volume_db = -20.0
	add_child(engine_player)
	engine_player.play()
	sfx_player = AudioStreamPlayer.new()
	sfx_player.volume_db = -8.0
	add_child(sfx_player)

func _engine_samples() -> PackedFloat32Array:
	var rate := 22050
	var samples := PackedFloat32Array()
	samples.resize(rate)
	for i in range(rate):
		var t := float(i) / float(rate)
		var hum := sin(TAU * 55.0 * t) * 0.5 + sin(TAU * 110.0 * t) * 0.22 + sin(TAU * 165.0 * t) * 0.08
		var wobble := 0.85 + 0.15 * sin(TAU * 2.0 * t)
		samples[i] = hum * wobble * 0.35
	return samples

func _noise_samples(duration: float, decay: float, tone_hz: float = 0.0) -> PackedFloat32Array:
	var rate := 22050
	var count := int(rate * duration)
	var samples := PackedFloat32Array()
	samples.resize(count)
	for i in range(count):
		var t := float(i) / float(rate)
		var envelope := pow(maxf(0.0, 1.0 - t / duration), decay)
		var noise := randf_range(-1.0, 1.0) * 0.6
		var tone := sin(TAU * tone_hz * t) * 0.4 if tone_hz > 0.0 else 0.0
		samples[i] = (noise + tone) * envelope
	return samples

func _make_wav(samples: PackedFloat32Array, looped: bool) -> AudioStreamWAV:
	var data := PackedByteArray()
	data.resize(samples.size() * 2)
	for i in range(samples.size()):
		data.encode_s16(i * 2, int(clampf(samples[i], -1.0, 1.0) * 32767.0))
	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = 22050
	stream.stereo = false
	stream.data = data
	if looped:
		stream.loop_mode = AudioStreamWAV.LOOP_FORWARD
		stream.loop_begin = 0
		stream.loop_end = samples.size()
	return stream

func _play_sfx(stream: AudioStreamWAV) -> void:
	if sfx_player != null and stream != null:
		sfx_player.stream = stream
		sfx_player.play()

func _update_audio() -> void:
	if engine_player == null:
		return
	var active := state == "playing" or state == "countdown"
	if not active:
		engine_player.volume_db = -60.0
		return
	engine_player.volume_db = -20.0
	engine_player.pitch_scale = 0.65 + clampf(sim.player_speed / maxf(1.0, sim.max_speed), 0.0, 1.0) * 0.7

func _update_ai_panel(action: String, label_source: String, metadata: Dictionary) -> void:
	if ai_model_label == null:
		return
	var model_text := String(metadata.get("model_alias", metadata.get("model_id", label_source)))
	if label_source == "controller_fallback":
		model_text = "FALLBACK"
	elif label_source == "human":
		model_text = "HUMAN CONTROL"
	ai_model_label.text = "AI  " + model_text
	var probabilities = metadata.get("probabilities", {})
	var probability := 0.0
	if typeof(probabilities) == TYPE_DICTIONARY:
		probability = float(probabilities.get(action, metadata.get("confidence", 0.0)))
	elif metadata.has("confidence"):
		probability = float(metadata.get("confidence", 0.0))
	ai_action_label.text = "ACTION  %s   %d%%" % [action, int(round(probability * 100.0))]
	var latency := float(metadata.get("model_latency_ms", metadata.get("client_latency_ms", 0.0)))
	ai_latency_label.text = "LATENCY  %.0f ms" % latency

func _load_highscore() -> void:
	var f := FileAccess.open(highscore_path, FileAccess.READ)
	if f:
		best_score = float(f.get_line())
		f.close()

func _save_highscore() -> void:
	var f := FileAccess.open(highscore_path, FileAccess.WRITE)
	if f:
		f.store_line(str(best_score))
		f.close()

func _start_auto_run() -> void:
	auto_running = true
	auto_episode_index = 0
	start_recording()
	auto_total_episodes = auto_seeds.size() * auto_episodes_per_seed
	if auto_max_episodes > 0:
		auto_total_episodes = min(auto_total_episodes, auto_max_episodes)
	_auto_start_next()

func _auto_start_next() -> void:
	if auto_episode_index >= auto_total_episodes:
		auto_running = false
		_finish_auto_run()
		return
	var seed_idx := auto_episode_index / auto_episodes_per_seed
	var seed: int = auto_seeds[seed_idx % auto_seeds.size()]
	var cfg := DEFAULT_CONFIG.duplicate(true)
	if auto_episode_duration > 0.0:
		cfg["episode_duration"] = auto_episode_duration
	if traffic_density_override >= 0.0:
		cfg["traffic_density"] = traffic_density_override
	external_control = true
	control_mode = "auto"
	start_game(seed, cfg)
	sim.control_source = "external_ai"

func _auto_next() -> void:
	auto_episode_index += 1
	_auto_start_next()

func _finish_auto_run() -> void:
	await get_tree().create_timer(0.25, true).timeout
	telemetry.close()
	get_tree().quit()

func _exit_tree() -> void:
	if telemetry != null:
		telemetry.close()

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if sim != null and sim.status == "running":
			sim.finish_episode("window_close")
		if telemetry != null:
			telemetry.close()
		get_tree().quit()

func get_session_dir() -> String:
	return telemetry.get_session_path()
