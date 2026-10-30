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

# 2D presentation constants. World units are metres; the simulation is unchanged.
# X and Z use separate pixel scales: the road stays wide and readable while the
# top-down view still shows enough distance ahead to see traffic coming.
const LANE_PX := 120.0
const ROAD_HALF_PX := 180.0
const PX_PER_M_Z := 10.0
const DASH_LENGTH := 26.0
const DASH_GAP := 24.0
const PLAYER_SCREEN_Y := 0.78

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
var notice_timer: float = 0.0

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
var action_event_sequence: int = 0
var external_timeout_msec: int = 1500
var max_stale_ticks: int = 30
var traffic_density_override: float = -1.0
var last_external_action_msec: int = 0
var websocket_port: int = 8765
var decision_state_history: Dictionary = {}
var decision_state_history_seconds: float = 180.0
var last_score_milestone: int = 0

var build_commit: String = "dev"
var build_godot_version: String = ""
var shake_timer: float = 0.0
var shake_intensity: float = 0.0
var shake_offset: Vector2 = Vector2.ZERO

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
	telemetry = TelemetryLogger.new()
	add_child(telemetry)
	net = NetBridge.new()
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
		"ws_port": 8765,
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
	var center_x := get_viewport_rect().size.x * 0.5
	return center_x - (world_x / 4.0) * LANE_PX

func _draw() -> void:
	if sim == null:
		return
	var view := get_viewport_rect().size
	var center_x := view.x * 0.5
	var player_y := view.y * PLAYER_SCREEN_Y
	var road_half := ROAD_HALF_PX
	var lane_px := LANE_PX

	draw_set_transform(shake_offset, 0.0, Vector2.ONE)

	# grass background with soft scrolling bands
	draw_rect(Rect2(Vector2.ZERO, view), Color(0.075, 0.135, 0.085))
	var band_offset := fmod(sim.player_z * PX_PER_M_Z, 96.0)
	var by := -96.0 + band_offset
	while by < view.y:
		draw_rect(Rect2(0, by, view.x, 48), Color(0.085, 0.155, 0.095))
		by += 96.0

	# road
	draw_rect(Rect2(center_x - road_half, 0, road_half * 2.0, view.y), Color(0.115, 0.125, 0.14))
	draw_rect(Rect2(center_x - road_half - 8, 0, 8, view.y), Color(0.72, 0.74, 0.76))
	draw_rect(Rect2(center_x + road_half, 0, 8, view.y), Color(0.72, 0.74, 0.76))

	# dashed lane dividers, scrolling toward the player
	var period := DASH_LENGTH + DASH_GAP
	var dash_offset := fmod(sim.player_z * PX_PER_M_Z, period)
	var dy := -period + dash_offset
	for lane_x in [center_x - lane_px, center_x + lane_px]:
		var y := dy
		while y < view.y + period:
			draw_rect(Rect2(lane_x - 2, y, 4, DASH_LENGTH), Color(0.85, 0.88, 0.9, 0.8))
			y += period

	# roadside posts
	var post_start: float = floor(sim.player_z / 24.0) * 24.0
	for i in range(-2, 14):
		var z: float = post_start + i * 24.0
		var py: float = player_y - (z - sim.player_z) * PX_PER_M_Z
		if py < -40 or py > view.y + 40:
			continue
		for side in [-1, 1]:
			var px: float = center_x + side * (road_half + 30.0)
			draw_rect(Rect2(px - 4, py - 20, 8, 20), Color(0.5, 0.53, 0.57))
			draw_rect(Rect2(px - 6, py - 22, 12, 4), Color(0.78, 0.8, 0.82))

	# traffic
	for v in sim.traffic:
		_draw_vehicle(v, center_x, player_y)

	# player
	_draw_bike(center_x - (sim.player_x / 4.0) * LANE_PX, player_y)

	# speed streaks at the edges
	var speed_ratio := clampf((sim.player_speed - sim.base_speed) / maxf(1.0, sim.max_speed - sim.base_speed), 0.0, 1.0)
	if speed_ratio > 0.05:
		var streak_alpha := 0.10 + speed_ratio * 0.22
		for i in range(7):
			var sx := center_x - road_half - 22 - (i % 3) * 14
			var sy := fmod(sim.player_z * PX_PER_M_Z * 1.8 + i * 97.0, view.y)
			draw_rect(Rect2(sx, sy, 3, 40 + speed_ratio * 50), Color(0.7, 0.9, 1.0, streak_alpha))
			draw_rect(Rect2(center_x + road_half + 22 + (i % 3) * 14, sy, 3, 40 + speed_ratio * 50), Color(0.7, 0.9, 1.0, streak_alpha))

	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)

func _draw_vehicle(v: Dictionary, center_x: float, player_y: float) -> void:
	var vz := float(v.get("z", 0.0))
	var dy := player_y - (vz - sim.player_z) * PX_PER_M_Z
	if dy < -260 or dy > get_viewport_rect().size.y + 260:
		return
	var lane := int(v.get("lane", 1))
	var vx := center_x - (sim.lane_center(lane) / 4.0) * LANE_PX
	var is_truck := String(v.get("type", "car")) == "truck"
	var length_px := float(v.get("length", 4.5)) * 16.0
	var width_px := float(v.get("width", 1.8)) * 26.0
	var color := Color(0.86, 0.2, 0.18) if not is_truck else Color(0.86, 0.58, 0.12)
	var body := Rect2(vx - width_px * 0.5, dy - length_px * 0.5, width_px, length_px)
	# soft shadow
	draw_rect(Rect2(body.position + Vector2(4, 6), body.size), Color(0, 0, 0, 0.28))
	draw_rect(body, color)
	draw_rect(Rect2(body.position.x + 3, body.position.y + 3, body.size.x - 6, body.size.y - 6), color.darkened(0.18))
	# windshield / roof
	var roof_h := body.size.y * 0.28
	draw_rect(Rect2(body.position.x + 5, body.position.y + body.size.y * 0.18, body.size.x - 10, roof_h), Color(0.12, 0.18, 0.24))
	# taillights
	draw_rect(Rect2(body.position.x + 4, body.end.y - 6, 7, 4), Color(1.0, 0.15, 0.1))
	draw_rect(Rect2(body.end.x - 11, body.end.y - 6, 7, 4), Color(1.0, 0.15, 0.1))
	# headlights
	draw_rect(Rect2(body.position.x + 4, body.position.y + 2, 7, 3), Color(1.0, 0.95, 0.7, 0.8))
	draw_rect(Rect2(body.end.x - 11, body.position.y + 2, 7, 3), Color(1.0, 0.95, 0.7, 0.8))

func _draw_bike(bx: float, by: float) -> void:
	var lean := clampf(-sim.player_velocity_x * 0.035, -0.35, 0.35)
	# shadow, drawn without lean so it stays under the bike
	draw_set_transform(Vector2(bx + 4, by + 8) + shake_offset, 0.0, Vector2.ONE)
	draw_rect(Rect2(-17, -44, 34, 88), Color(0, 0, 0, 0.28))
	draw_set_transform(Vector2(bx, by) + shake_offset, lean, Vector2.ONE)
	# wheels
	draw_rect(Rect2(-6, -44, 12, 24), Color(0.05, 0.05, 0.06))
	draw_rect(Rect2(-6, 20, 12, 24), Color(0.05, 0.05, 0.06))
	draw_rect(Rect2(-6, -42, 12, 4), Color(0.28, 0.3, 0.33))
	draw_rect(Rect2(-6, 38, 12, 4), Color(0.28, 0.3, 0.33))
	# frame and tank
	draw_rect(Rect2(-10, -20, 20, 46), Color(0.12, 0.45, 0.72))
	draw_rect(Rect2(-9, -8, 18, 20), Color(0.84, 0.2, 0.15))
	draw_rect(Rect2(-7, -6, 14, 5), Color(1.0, 0.45, 0.3, 0.6))
	# handlebars
	draw_rect(Rect2(-22, -26, 44, 5), Color(0.65, 0.68, 0.72))
	# rider
	draw_circle(Vector2(0, 2), 13, Color(0.14, 0.16, 0.2))
	draw_circle(Vector2(0, -13), 10, Color(0.9, 0.9, 0.88))
	draw_rect(Rect2(-7, -16, 14, 4), Color(0.15, 0.55, 0.75))
	# headlight
	draw_rect(Rect2(-5, -46, 10, 4), Color(1.0, 0.95, 0.65))
	if sim.braking:
		draw_circle(Vector2(0, 44), 7, Color(1.0, 0.1, 0.05, 0.9))
	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

func _build_ui() -> void:
	ui_layer = CanvasLayer.new()
	ui_layer.process_mode = Node.PROCESS_MODE_ALWAYS
	add_child(ui_layer)
	hud = Control.new()
	hud.set_anchors_preset(Control.PRESET_FULL_RECT)
	ui_layer.add_child(hud)
	var ui_theme := Theme.new()
	ui_theme.set_color("font_color", "Label", Color(0.87, 0.94, 0.98))
	ui_theme.set_color("font_color", "Button", Color(0.9, 0.97, 1.0))
	ui_theme.set_font_size("font_size", "Label", 16)
	ui_theme.set_font_size("font_size", "Button", 18)
	ui_theme.set_constant("separation", "HBoxContainer", 24)
	ui_theme.set_constant("separation", "VBoxContainer", 12)
	var panel_style := StyleBoxFlat.new()
	panel_style.bg_color = Color(0.025, 0.055, 0.09, 0.95)
	panel_style.border_color = Color(0.1, 0.78, 0.85, 0.9)
	panel_style.set_border_width_all(1)
	panel_style.set_corner_radius_all(12)
	panel_style.content_margin_left = 22
	panel_style.content_margin_right = 22
	panel_style.content_margin_top = 18
	panel_style.content_margin_bottom = 18
	ui_theme.set_stylebox("panel", "PanelContainer", panel_style)
	var button_style := StyleBoxFlat.new()
	button_style.bg_color = Color(0.05, 0.16, 0.2)
	button_style.set_corner_radius_all(6)
	button_style.content_margin_left = 12
	button_style.content_margin_right = 12
	button_style.content_margin_top = 8
	button_style.content_margin_bottom = 8
	var button_hover := button_style.duplicate()
	button_hover.bg_color = Color(0.08, 0.34, 0.39)
	var button_pressed := button_style.duplicate()
	button_pressed.bg_color = Color(0.04, 0.5, 0.55)
	ui_theme.set_stylebox("normal", "Button", button_style)
	ui_theme.set_stylebox("hover", "Button", button_hover)
	ui_theme.set_stylebox("pressed", "Button", button_pressed)
	hud.theme = ui_theme

	var top_panel := PanelContainer.new()
	top_panel.set_anchors_preset(Control.PRESET_TOP_WIDE)
	top_panel.offset_left = 20
	top_panel.offset_top = 14
	top_panel.offset_right = -20
	top_panel.offset_bottom = 60
	hud.add_child(top_panel)
	var top := HBoxContainer.new()
	top.alignment = BoxContainer.ALIGNMENT_CENTER
	top_panel.add_child(top)
	score_label = Label.new()
	score_label.text = "Score: 0"
	top.add_child(score_label)
	dist_label = Label.new()
	dist_label.text = "Dist: 0"
	top.add_child(dist_label)
	speed_label = Label.new()
	speed_label.text = "Speed: 0"
	top.add_child(speed_label)
	best_label = Label.new()
	best_label.text = "Best: 0"
	top.add_child(best_label)

	record_button = Button.new()
	record_button.text = "Record"
	record_button.custom_minimum_size = Vector2(90, 32)
	record_button.pressed.connect(_on_record_button_pressed)
	top.add_child(record_button)
	record_status_label = Label.new()
	record_status_label.text = ""
	top.add_child(record_status_label)

	status_label = Label.new()
	status_label.text = ""
	status_label.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	status_label.offset_left = -280
	status_label.offset_top = -52
	status_label.offset_right = -24
	status_label.offset_bottom = -48
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	hud.add_child(status_label)

	conn_label = Label.new()
	conn_label.text = "WS: disconnected"
	conn_label.set_anchors_preset(Control.PRESET_BOTTOM_LEFT)
	hud.add_child(conn_label)

	msg_label = Label.new()
	msg_label.text = "A / LEFT: MOVE LEFT     D / RIGHT: MOVE RIGHT     SPACE / DOWN: BRAKE     T: SWITCH CONTROL     ESC: PAUSE     F9: RECORD"
	msg_label.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	msg_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hud.add_child(msg_label)

	notice_label = Label.new()
	notice_label.set_anchors_preset(Control.PRESET_TOP_WIDE)
	notice_label.offset_top = 72
	notice_label.offset_bottom = 104
	notice_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	notice_label.add_theme_color_override("font_color", Color(0.4, 0.95, 0.85))
	notice_label.add_theme_font_size_override("font_size", 22)
	notice_label.visible = false
	hud.add_child(notice_label)

	menu_panel = _make_panel("LANESHIFT 2D", ["Start", "Open Logs", "Quit"])
	menu_panel.theme = ui_theme
	ui_layer.add_child(menu_panel)
	pause_panel = _make_panel("PAUSED", ["Resume", "Restart", "Menu"])
	pause_panel.theme = ui_theme
	pause_panel.visible = false
	ui_layer.add_child(pause_panel)
	gameover_panel = _make_panel("GAME OVER", ["Restart", "Menu"])
	gameover_panel.theme = ui_theme
	gameover_panel.visible = false
	ui_layer.add_child(gameover_panel)
	gameover_result_label = gameover_panel.find_child("Result", true, false) as Label
	if gameover_result_label == null:
		gameover_result_label = Label.new()
		gameover_result_label.name = "Result"
		gameover_result_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		(gameover_panel.get_child(0) as VBoxContainer).add_child(gameover_result_label)

	_wire_menu()

func _make_panel(title: String, buttons: Array) -> Control:
	var panel := PanelContainer.new()
	var panel_size := Vector2(340, 260)
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
	panel.add_child(vbox)
	var t := Label.new()
	t.text = title
	t.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	t.add_theme_color_override("font_color", Color(0.25, 0.91, 0.94))
	t.add_theme_font_size_override("font_size", 28)
	vbox.add_child(t)
	if title == "LANESHIFT 2D":
		var tagline := Label.new()
		tagline.text = "THREE LANES. NO SECOND CHANCES."
		tagline.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		tagline.add_theme_color_override("font_color", Color(0.65, 0.76, 0.81))
		tagline.add_theme_font_size_override("font_size", 12)
		vbox.add_child(tagline)
	for bname in buttons:
		var btn := Button.new()
		btn.text = bname
		btn.name = bname
		btn.custom_minimum_size = Vector2(220, 42)
		vbox.add_child(btn)
		if title == "GAME OVER" and bname == "Restart":
			var result := Label.new()
			result.name = "Result"
			result.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
			result.text = ""
			vbox.add_child(result)
	return panel

func _wire_menu() -> void:
	for btn in _find_buttons(menu_panel):
		btn.pressed.connect(func(): _on_menu_button(btn.text))
	for btn in _find_buttons(pause_panel):
		btn.pressed.connect(func(): _on_pause_button(btn.text))
	for btn in _find_buttons(gameover_panel):
		btn.pressed.connect(func(): _on_gameover_button(btn.text))

func _find_buttons(root: Node) -> Array:
	var res := []
	for c in root.get_children():
		if c is Button:
			res.append(c)
		res += _find_buttons(c)
	return res

func _on_menu_button(text: String) -> void:
	if text == "Start":
		start_game()
	elif text == "Open Logs":
		var runs_path := PortablePathsScript.get_recordings_root()
		DirAccess.make_dir_recursive_absolute(runs_path)
		OS.shell_open(runs_path)
	elif text == "Quit":
		if sim.status == "running":
			sim.finish_episode("quit")
		telemetry.close()
		get_tree().quit()

func _on_pause_button(text: String) -> void:
	if text == "Resume":
		resume_game()
	elif text == "Restart":
		restart_game()
	elif text == "Menu":
		go_menu()

func _on_gameover_button(text: String) -> void:
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
	menu_panel.visible = false
	pause_panel.visible = false
	gameover_panel.visible = false
	state = "playing"
	telemetry_timer = 0.0
	decision_timer = 0.0
	queue_redraw()

func restart_game() -> void:
	start_game(rng_visual.randi())

func go_menu() -> void:
	if sim.status == "running":
		sim.finish_episode("menu_exit")
	get_tree().paused = false
	state = "menu"
	menu_panel.visible = true
	pause_panel.visible = false
	gameover_panel.visible = false
	queue_redraw()

func _on_crashed(_eid: String) -> void:
	shake_timer = 0.6
	shake_intensity = 0.6
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
		state = "gameover"
		gameover_panel.visible = true
		gameover_result_label.text = "Score %.0f   Distance %.0f m\nSurvived %.1f s" % [summary["score"], summary["distance"], summary["survival_time"]]

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

func _on_net_connected() -> void:
	conn_label.text = "WS: connected"
	_log_event("controller_connected", {})

func _on_net_disconnected() -> void:
	conn_label.text = "WS: disconnected"
	if not auto_run:
		external_control = false
		control_mode = "human"
	_log_event("controller_disconnected", {})

func _submit_action(action: String, duration: float, control_source: String, label_source: String, controller_seq: int = -1, state_tick: int = -1, decision_state: Dictionary = {}, decision_metadata: Dictionary = {}) -> bool:
	var application_state: Dictionary = sim.get_snapshot()
	var state_before: Dictionary = decision_state.duplicate(true) if not decision_state.is_empty() else application_state
	var decision_tick := int(state_before.get("physics_tick", state_tick if state_tick >= 0 else sim.physics_tick))
	var application_tick := sim.physics_tick
	var accepted: bool = sim.apply_action(action, duration, control_source)
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
	telemetry.tick(delta)
	if notice_timer > 0.0:
		notice_timer = max(0.0, notice_timer - delta)
		notice_label.visible = notice_timer > 0.0
	if shake_timer > 0:
		shake_timer -= delta
		if shake_timer <= 0:
			shake_offset = Vector2.ZERO
	if state == "playing":
		_handle_human_input(delta)
		if external_control and not auto_run and Time.get_ticks_msec() - last_external_action_msec > external_timeout_msec:
			external_control = false
			control_mode = "human"
			_log_event("controller_timeout", {"fallback_control": "human"})
			_submit_action("STAY", 0.1, "human", "human")
		sim.step(delta)
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
			sim.apply_action("BRAKE", 0.15, "human")
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
	if _key_just_pressed(KEY_F9):
		_toggle_recording()
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
	conn_label.text = "WS: " + ("connected" if net.connected_flag else "disconnected")
	if state == "playing" and take_control_on_start and net.connected_flag and not external_control:
		external_control = true
		control_mode = "external"
		last_external_action_msec = Time.get_ticks_msec()
		sim.control_source = "external_ai"
		take_control_on_start = false
		_log_event("control_mode_changed", {"control_mode": "external", "reason": "take_control_on_start"})
	_update_record_ui()
	if recording_active and record_status_label != null:
		var elapsed := float(Time.get_ticks_msec() - recording_start_msec) / 1000.0
		var samples := 0
		var events := 0
		if telemetry != null:
			samples = telemetry.telemetry_record_count
			events = telemetry.event_record_count
		record_status_label.text = "REC %.1fs  samples:%d events:%d" % [elapsed, samples, events]
	if state == "menu":
		menu_panel.visible = true
	if shake_timer > 0.0:
		shake_offset = Vector2(randf_range(-shake_intensity, shake_intensity), randf_range(-shake_intensity, shake_intensity)) * 4.0

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
	state = "paused"
	get_tree().paused = true
	pause_panel.visible = true

func resume_game() -> void:
	if state != "paused":
		return
	_log_event("resume", {})
	state = "playing"
	get_tree().paused = false
	pause_panel.visible = false

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
	if telemetry == null or (telemetry != null and telemetry.writer_closed):
		telemetry = TelemetryLogger.new()
		add_child(telemetry)
	telemetry.build_commit = build_commit
	telemetry.build_godot_version = build_godot_version
	telemetry.start_session(PortablePathsScript.get_recordings_root())
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
	_toggle_recording()

func _update_record_ui() -> void:
	if record_button == null:
		return
	if recording_active:
		record_button.text = "Stop"
		record_status_label.text = "REC"
	else:
		record_button.text = "Record"
		record_status_label.text = ""

func _update_hud() -> void:
	score_label.text = "Score: %.0f" % sim.score
	dist_label.text = "Dist: %.0f m" % sim.distance
	speed_label.text = "Speed: %.0f" % (sim.player_speed * 3.6)
	best_label.text = "Best: %.0f" % best_score
	status_label.text = "CONTROL  " + control_mode.to_upper()

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
