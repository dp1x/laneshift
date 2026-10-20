extends Node3D

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
const GROUND_SEGMENT_LENGTH := 3000.0
const GROUND_SEGMENT_COUNT := 4
const GROUND_RECYCLE_MARGIN := 250.0
const ROAD_SEGMENT_LENGTH := 50.0
const ROAD_SEGMENT_COUNT := 16
const ROAD_WIDTH := 12.0
const ROAD_TILE_LENGTH := 12.0

var sim: Simulation
var telemetry: TelemetryLogger
var net: NetBridge

var camera: Camera3D
var player_node: Node3D
var traffic_root: Node3D
var road_root: Node3D
var tree_root: Node3D
var ground_segments: Array[MeshInstance3D] = []
var blob_shadow_texture: GradientTexture2D = null
var road_texture: ImageTexture = null
var grass_texture: ImageTexture = null

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
var notice_label: Label
var tail_light: MeshInstance3D
var notice_timer: float = 0.0

var best_score: float = 0.0
var highscore_path := "user://highscore.save"

var telemetry_timer: float = 0.0
var telemetry_hz: float = 30.0
var decision_timer: float = 0.0
var decision_hz: float = 10.0
var net_send_timer: float = 0.0

var control_mode: String = "human" # human, auto
var external_control: bool = false

var auto_run: bool = false
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

var road_segments: Array = []
var tree_nodes: Array = []
var hill_nodes: Array = []
var cloud_nodes: Array = []
var traffic_nodes: Dictionary = {}

var shake_timer: float = 0.0
var shake_intensity: float = 0.0

var brake_held: bool = false

var rng_visual: RandomNumberGenerator = RandomNumberGenerator.new()

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	rng_visual.randomize()
	_load_highscore()
	_parse_args()
	sim = Simulation.new()
	add_child(sim)
	sim.configure(DEFAULT_CONFIG.duplicate())
	sim.crashed.connect(_on_crashed)
	sim.episode_ended.connect(_on_episode_ended)
	sim.scenario_initialized.connect(_on_scenario_init)
	sim.vehicle_overtaken.connect(_on_vehicle_overtaken)
	_build_world()
	_build_ui()
	telemetry = TelemetryLogger.new()
	add_child(telemetry)
	telemetry.start_session()
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

func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		args = OS.get_cmdline_args()
	var options := parse_options(args)
	auto_run = bool(options["auto_run"])
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

## Parses "--key=value" options. Splitting on the first "=" keeps the value
## intact, so an option can never be silently mis-read.
static func parse_options(args: PackedStringArray) -> Dictionary:
	var options := {
		"auto_run": false,
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

func _build_world() -> void:
	# lighting
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-45, -30, 0)
	# Real-time shadow maps are the single most expensive effect on a machine
	# without a GPU, so shadows are drawn as cheap blob decals instead.
	sun.shadow_enabled = false
	sun.light_color = Color(1.0, 0.96, 0.87)
	sun.light_energy = 1.25
	add_child(sun)
	var env := WorldEnvironment.new()
	var e := Environment.new()
	# The compatibility renderer has no depth fog, so the sky carries the haze:
	# its horizon band is tinted to meet the ground colour.
	var sky_material := ProceduralSkyMaterial.new()
	sky_material.sky_top_color = Color(0.35, 0.55, 0.88)
	sky_material.sky_horizon_color = Color(0.74, 0.83, 0.86)
	sky_material.ground_horizon_color = Color(0.66, 0.76, 0.76)
	sky_material.ground_bottom_color = Color(0.22, 0.32, 0.28)
	sky_material.sun_angle_max = 40.0
	var sky := Sky.new()
	sky.sky_material = sky_material
	e.background_mode = Environment.BG_SKY
	e.sky = sky
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.6, 0.65, 0.7)
	e.ambient_light_energy = 0.4
	env.environment = e
	add_child(env)

	# road root
	road_root = Node3D.new()
	add_child(road_root)
	var ground_material := StandardMaterial3D.new()
	ground_material.albedo_color = Color(1, 1, 1)
	ground_material.albedo_texture = _grass_texture()
	ground_material.uv1_scale = Vector3(26.0, 60.0, 1.0)
	for i in range(GROUND_SEGMENT_COUNT):
		var ground := MeshInstance3D.new()
		ground.name = "GroundSegment%d" % i
		var ground_mesh := BoxMesh.new()
		ground_mesh.size = Vector3(400, 0.2, GROUND_SEGMENT_LENGTH)
		ground.mesh = ground_mesh
		ground.material_override = ground_material
		ground.position = Vector3(0, -0.32, 120.0 + i * GROUND_SEGMENT_LENGTH)
		road_root.add_child(ground)
		ground_segments.append(ground)
	_build_road()

	# trees
	tree_root = Node3D.new()
	add_child(tree_root)
	_build_trees()

	# traffic root
	traffic_root = Node3D.new()
	add_child(traffic_root)

	# player
	player_node = _create_motorcycle()
	player_node.scale = Vector3.ONE * 1.15
	add_child(player_node)

	# camera
	camera = Camera3D.new()
	camera.fov = 60
	add_child(camera)
	_update_camera(true)

func _build_road() -> void:
	for i in range(ROAD_SEGMENT_COUNT):
		var seg := _create_road_segment()
		seg.position.z = i * ROAD_SEGMENT_LENGTH
		road_root.add_child(seg)
		road_segments.append(seg)

## Builds the road surface, markings and guard rails procedurally. The lane
## markings are painted into a repeating texture rather than modelled, which
## keeps long stretches of road cheap to draw.
func _create_road_segment() -> Node3D:
	var seg := Node3D.new()
	var mesh := BoxMesh.new()
	mesh.size = Vector3(ROAD_WIDTH, 0.2, ROAD_SEGMENT_LENGTH)
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	var mat := StandardMaterial3D.new()
	# The texture already carries the surface colour, so the albedo stays white
	# instead of darkening it a second time.
	mat.albedo_color = Color(1, 1, 1)
	mat.albedo_texture = _road_texture()
	mat.uv1_scale = Vector3(1.0, ROAD_SEGMENT_LENGTH / ROAD_TILE_LENGTH, 1.0)
	mi.material_override = mat
	mi.position.y = -0.1
	seg.add_child(mi)

	# guard rails along both edges
	for side in [-1, 1]:
		var rail := MeshInstance3D.new()
		var rail_mesh := BoxMesh.new()
		rail_mesh.size = Vector3(0.12, 0.3, ROAD_SEGMENT_LENGTH)
		rail.mesh = rail_mesh
		var rail_material := StandardMaterial3D.new()
		rail_material.albedo_color = Color(0.72, 0.74, 0.78)
		rail_material.metallic = 0.6
		rail_material.roughness = 0.45
		rail.material_override = rail_material
		rail.position = Vector3(side * 6.35, 0.55, 0)
		seg.add_child(rail)
		# posts
		for i in range(ROAD_SEGMENT_LENGTH / 12.5):
			var post := MeshInstance3D.new()
			var post_mesh := BoxMesh.new()
			post_mesh.size = Vector3(0.1, 0.9, 0.1)
			post.mesh = post_mesh
			var post_material := StandardMaterial3D.new()
			post_material.albedo_color = Color(0.5, 0.52, 0.55)
			post.material_override = post_material
			post.position = Vector3(side * 6.35, 0.3, -ROAD_SEGMENT_LENGTH * 0.5 + 6.0 + i * 12.5)
			seg.add_child(post)
	return seg

## Paints one repeating tile of road: asphalt grain, solid white edge lines
## and dashed yellow lane boundaries, at the exact lane geometry.
func _road_texture() -> ImageTexture:
	if road_texture != null:
		return road_texture
	var size := 256
	var image := Image.create_empty(size, size, false, Image.FORMAT_RGB8)
	var rng := RandomNumberGenerator.new()
	rng.seed = 20240517
	for y in range(size):
		for x in range(size):
			var u := float(x) / float(size)
			var v := float(y) / float(size)
			var grain := rng.randf_range(-0.04, 0.04)
			var color := Color(0.30 + grain, 0.31 + grain, 0.34 + grain)
			# solid white edge lines
			if u < 0.035 or u > 0.965:
				color = Color(0.92, 0.92, 0.90)
			# dashed yellow lane boundaries: 3m dash inside a 12m tile
			var dash := fmod(v * 4.0, 1.0) < 0.25
			if dash and (u > 0.315 and u < 0.352 or u > 0.648 and u < 0.685):
				color = Color(0.98, 0.80, 0.14)
			image.set_pixel(x, y, color)
	road_texture = ImageTexture.create_from_image(image)
	return road_texture

func _grass_texture() -> ImageTexture:
	if grass_texture != null:
		return grass_texture
	var size := 128
	var image := Image.create_empty(size, size, false, Image.FORMAT_RGB8)
	var rng := RandomNumberGenerator.new()
	rng.seed = 991
	for y in range(size):
		for x in range(size):
			var n := rng.randf_range(-0.05, 0.05)
			var patch := sin(float(x) * 0.21) * cos(float(y) * 0.17) * 0.03
			image.set_pixel(x, y, Color(0.27 + n + patch, 0.44 + n + patch, 0.29 + n + patch))
	grass_texture = ImageTexture.create_from_image(image)
	return grass_texture

func _build_trees() -> void:
	for i in range(22):
		var t := _create_tree()
		var side := -1 if i % 2 == 0 else 1
		t.position = Vector3(side * rng_visual.randf_range(9.0, 16.0), 0, i * 16.0 - 60.0)
		tree_root.add_child(t)
		tree_nodes.append(t)
	# distant scenery that scrolls with the road
	for i in range(6):
		var hill := _create_hill()
		var side := -1 if i % 2 == 0 else 1
		hill.position = Vector3(side * rng_visual.randf_range(90.0, 220.0), 0, i * 260.0 - 200.0)
		tree_root.add_child(hill)
		hill_nodes.append(hill)
	for i in range(7):
		var cloud := _create_cloud()
		cloud.position = Vector3(rng_visual.randf_range(-160.0, 160.0), rng_visual.randf_range(55.0, 95.0), i * 220.0 - 150.0)
		tree_root.add_child(cloud)
		cloud_nodes.append(cloud)

func _add_blob_shadow(parent: Node3D, size: Vector2, offset: Vector3 = Vector3.ZERO) -> void:
	var shadow := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = size
	shadow.mesh = plane
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.05, 0.07, 0.06, 0.5)
	material.albedo_texture = _blob_shadow_texture()
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	shadow.material_override = material
	shadow.position = Vector3(offset.x, 0.02, offset.z)
	shadow.rotation_degrees = Vector3(-90, 0, 0)
	parent.add_child(shadow)

func _blob_shadow_texture() -> GradientTexture2D:
	if blob_shadow_texture == null:
		var gradient := Gradient.new()
		gradient.set_color(0, Color(0, 0, 0, 0.85))
		gradient.set_color(1, Color(0, 0, 0, 0))
		blob_shadow_texture = GradientTexture2D.new()
		blob_shadow_texture.gradient = gradient
		blob_shadow_texture.fill = GradientTexture2D.FILL_RADIAL
		blob_shadow_texture.fill_from = Vector2(0.5, 0.5)
		blob_shadow_texture.fill_to = Vector2(1.0, 0.5)
		blob_shadow_texture.width = 64
		blob_shadow_texture.height = 64
	return blob_shadow_texture

func _create_tree() -> Node3D:
	var t := Node3D.new()
	var trunk := MeshInstance3D.new()
	var cm := CylinderMesh.new()
	cm.top_radius = 0.16
	cm.bottom_radius = 0.34
	cm.height = 2.4
	cm.radial_segments = 6
	cm.rings = 1
	trunk.mesh = cm
	var tmat := StandardMaterial3D.new()
	tmat.albedo_color = Color(0.32, 0.21, 0.13)
	trunk.material_override = tmat
	trunk.position.y = 1.2
	t.add_child(trunk)
	# layered canopy: three overlapping crowns of slightly different greens
	var crowns := [
		[Vector3(0, 3.3, 0), 1.75, Color(0.16, 0.42, 0.19)],
		[Vector3(0.45, 2.75, 0.25), 1.25, Color(0.20, 0.50, 0.22)],
		[Vector3(-0.4, 2.95, -0.2), 1.1, Color(0.13, 0.36, 0.17)]
	]
	for crown in crowns:
		var foliage := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = float(crown[1])
		sm.height = float(crown[1]) * 1.7
		sm.radial_segments = 7
		sm.rings = 4
		foliage.mesh = sm
		var fmat := StandardMaterial3D.new()
		fmat.albedo_color = crown[2]
		foliage.material_override = fmat
		foliage.position = crown[0]
		t.add_child(foliage)
	_add_blob_shadow(t, Vector2(3.8, 3.8), Vector3(0.8, 0, 0.4))
	return t

## A low, wide mound far out on either side, to break the flat horizon.
func _create_hill() -> Node3D:
	var hill := Node3D.new()
	for i in range(3):
		var mound := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = rng_visual.randf_range(40.0, 70.0)
		sm.height = sm.radius * rng_visual.randf_range(0.5, 0.9)
		sm.radial_segments = 8
		sm.rings = 4
		mound.mesh = sm
		var material := StandardMaterial3D.new()
		material.albedo_color = Color(0.14, 0.26, 0.24)
		mound.material_override = material
		mound.position = Vector3(rng_visual.randf_range(-25, 25), -sm.height * 0.42, rng_visual.randf_range(-30, 30))
		hill.add_child(mound)
	return hill

## A soft cloud puff, drawn high and far away.
func _create_cloud() -> Node3D:
	var cloud := Node3D.new()
	for i in range(4):
		var puff := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = rng_visual.randf_range(8.0, 16.0)
		sm.height = sm.radius * 0.8
		sm.radial_segments = 7
		sm.rings = 3
		puff.mesh = sm
		var material := StandardMaterial3D.new()
		material.albedo_color = Color(0.99, 0.99, 1.0)
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		material.albedo_color.a = 0.92
		puff.material_override = material
		puff.position = Vector3(rng_visual.randf_range(-14, 14), rng_visual.randf_range(-2, 2), rng_visual.randf_range(-8, 8))
		puff.scale = Vector3(1.0, 0.55, 1.0)
		cloud.add_child(puff)
	return cloud

func _make_box(parent: Node3D, size: Vector3, color: Color, position: Vector3, rotation_deg: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = size
	node.mesh = mesh
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	node.material_override = material
	node.position = position
	node.rotation_degrees = rotation_deg
	parent.add_child(node)
	return node

func _create_motorcycle() -> Node3D:
	var root := Node3D.new()
	var paint := Color(0.09, 0.42, 0.72)
	var dark := Color(0.09, 0.10, 0.12)
	var metal := Color(0.62, 0.65, 0.68)

	# wheels: front wheel slightly larger and further forward
	for wz in [-0.72, 0.78]:
		var wheel := MeshInstance3D.new()
		var wm := CylinderMesh.new()
		wm.top_radius = 0.36
		wm.bottom_radius = 0.36
		wm.height = 0.18
		wm.radial_segments = 12
		wm.rings = 1
		wheel.mesh = wm
		var wmat := StandardMaterial3D.new()
		wmat.albedo_color = dark
		wheel.material_override = wmat
		wheel.rotation_degrees = Vector3(0, 0, 90)
		wheel.position = Vector3(0, 0.36, wz)
		root.add_child(wheel)
		# hub
		_make_box(root, Vector3(0.2, 0.1, 0.1), metal, Vector3(0, 0.36, wz))

	# engine and frame block, tapering toward the tank
	_make_box(root, Vector3(0.42, 0.42, 1.0), dark, Vector3(0, 0.52, -0.05))
	_make_box(root, Vector3(0.5, 0.34, 0.62), paint, Vector3(0, 0.72, 0.18))
	# fuel tank
	_make_box(root, Vector3(0.44, 0.28, 0.7), paint, Vector3(0, 0.92, 0.18), Vector3(-6, 0, 0))
	# seat
	_make_box(root, Vector3(0.36, 0.12, 0.66), dark, Vector3(0, 0.98, -0.34), Vector3(3, 0, 0))
	# tail
	_make_box(root, Vector3(0.3, 0.22, 0.3), paint, Vector3(0, 0.96, -0.72), Vector3(12, 0, 0))
	# front fairing
	_make_box(root, Vector3(0.36, 0.52, 0.28), paint, Vector3(0, 0.92, 0.66), Vector3(-24, 0, 0))
	# forks
	for side in [-1, 1]:
		_make_box(root, Vector3(0.07, 0.62, 0.07), metal, Vector3(side * 0.13, 0.62, 0.72), Vector3(-20, 0, 0))
	# handlebars
	_make_box(root, Vector3(0.66, 0.07, 0.07), metal, Vector3(0, 1.16, 0.56))
	# exhaust
	_make_box(root, Vector3(0.12, 0.12, 0.8), metal, Vector3(0.2, 0.38, -0.3), Vector3(0, 0, 0))

	# rider: torso leaning forward over the tank
	var rider := MeshInstance3D.new()
	var rm := CapsuleMesh.new()
	rm.radius = 0.24
	rm.height = 0.86
	rm.radial_segments = 8
	rm.rings = 2
	rider.mesh = rm
	var rmat := StandardMaterial3D.new()
	rmat.albedo_color = Color(0.78, 0.22, 0.18)
	rider.material_override = rmat
	rider.position = Vector3(0, 1.32, -0.02)
	rider.rotation_degrees = Vector3(-38, 0, 0)
	root.add_child(rider)
	# arms reaching the bars
	for side in [-1, 1]:
		_make_box(root, Vector3(0.12, 0.12, 0.62), rmat.albedo_color, Vector3(side * 0.2, 1.24, 0.3), Vector3(-26, 0, 0))
	# legs
	for side in [-1, 1]:
		_make_box(root, Vector3(0.16, 0.16, 0.5), dark, Vector3(side * 0.16, 0.78, -0.18), Vector3(24, 0, 0))
	# helmet with a dark visor
	var head := MeshInstance3D.new()
	var hm := SphereMesh.new()
	hm.radius = 0.21
	hm.height = 0.42
	hm.radial_segments = 10
	hm.rings = 5
	head.mesh = hm
	var hmat := StandardMaterial3D.new()
	hmat.albedo_color = Color(0.95, 0.93, 0.88)
	head.material_override = hmat
	head.position = Vector3(0, 1.62, 0.2)
	root.add_child(head)
	_make_box(root, Vector3(0.34, 0.12, 0.06), Color(0.06, 0.07, 0.09), Vector3(0, 1.62, 0.38))

	_add_blob_shadow(root, Vector2(1.1, 2.2))
	tail_light = MeshInstance3D.new()
	var light_mesh := BoxMesh.new()
	light_mesh.size = Vector3(0.26, 0.12, 0.06)
	tail_light.mesh = light_mesh
	var light_material := StandardMaterial3D.new()
	light_material.albedo_color = Color(0.85, 0.08, 0.05)
	light_material.emission_enabled = true
	light_material.emission = Color(0.7, 0.02, 0.01)
	tail_light.material_override = light_material
	tail_light.position = Vector3(0, 0.99, -0.87)
	root.add_child(tail_light)
	return root

func _create_car(color: Color, is_truck: bool) -> Node3D:
	var root := Node3D.new()
	var length := 4.5 if not is_truck else 7.0
	var height := 0.7 if not is_truck else 1.2
	var body := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(1.8, height, length)
	body.mesh = bm
	var bmat := StandardMaterial3D.new()
	bmat.albedo_color = color
	body.material_override = bmat
	body.position.y = height * 0.5 + 0.2
	root.add_child(body)
	# cabin
	var cabin := MeshInstance3D.new()
	var cm := BoxMesh.new()
	cm.size = Vector3(1.6, 0.6, length * 0.5)
	cabin.mesh = cm
	var cmat := StandardMaterial3D.new()
	cmat.albedo_color = color.darkened(0.2)
	cabin.material_override = cmat
	cabin.position = Vector3(0, height + 0.5, -length * 0.1)
	root.add_child(cabin)
	# wheels
	for wz in [-length*0.35, length*0.35]:
		for sx in [-0.85, 0.85]:
			var w := MeshInstance3D.new()
			var wm := CylinderMesh.new()
			wm.top_radius = 0.3
			wm.bottom_radius = 0.3
			wm.height = 0.2
			w.mesh = wm
			w.rotation_degrees = Vector3(0, 0, 90)
			w.position = Vector3(sx, 0.3, wz)
			var wmat := StandardMaterial3D.new()
			wmat.albedo_color = Color(0.05, 0.05, 0.05)
			w.material_override = wmat
			root.add_child(w)
	_add_blob_shadow(root, Vector2(2.2, 5.0 if not is_truck else 7.2))
	return root

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

	# compact glass stats bar
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
	msg_label.text = "A / LEFT ARROW: MOVE LEFT     D / RIGHT ARROW: MOVE RIGHT     SPACE / DOWN: BRAKE     T: SWITCH CONTROL     ESC: PAUSE"
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

	# menu panel
	menu_panel = _make_panel("LANESHIFT", ["Start", "Open Logs", "Quit"])
	menu_panel.theme = ui_theme
	ui_layer.add_child(menu_panel)
	# pause panel
	pause_panel = _make_panel("PAUSED", ["Resume", "Restart", "Menu"])
	pause_panel.theme = ui_theme
	pause_panel.visible = false
	ui_layer.add_child(pause_panel)
	# gameover panel
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

	# wire buttons
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
	if title == "LANESHIFT":
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
	# menu
	var start_btn := menu_panel.get_node("VBoxContainer/Start") if menu_panel.has_node("VBoxContainer/Start") else null
	# nodes are direct children of panel? We added vbox as child; button parent is vbox. Find by name recursively.
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
		var runs_path := ProjectSettings.globalize_path("user://runs")
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
	player_node.visible = true
	_reset_world_positions()
	_reset_traffic_nodes()
	_update_camera(true)
	telemetry_timer = 0.0
	decision_timer = 0.0

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
	# Retain by age, not by count: a slow model may answer long after the
	# state it saw, and that exact snapshot must still be available.
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
	# camera shake decay
	if shake_timer > 0:
		shake_timer -= delta
	# input handling
	if state == "playing":
		_handle_human_input(delta)
		if external_control and not auto_run and Time.get_ticks_msec() - last_external_action_msec > external_timeout_msec:
			external_control = false
			control_mode = "human"
			_log_event("controller_timeout", {"fallback_control": "human"})
			_submit_action("STAY", 0.1, "human", "human")
		sim.step(delta)
		_log_score_milestones(sim.score)
		_sync_traffic_nodes()
		_update_camera(false)
		if sim.status != "running":
			_update_hud()
			return
		# telemetry
		telemetry_timer += delta
		if telemetry_timer >= 1.0 / telemetry_hz:
			telemetry_timer = 0.0
			var snap: Dictionary = sim.get_snapshot()
			snap["wall_time_unix"] = Time.get_unix_time_from_system()
			snap["wall_time_monotonic_msec"] = Time.get_ticks_msec()
			telemetry.log_telemetry(snap)
		# decision/net send
		decision_timer += delta
		if decision_timer >= 1.0 / decision_hz:
			decision_timer = 0.0
			var snap2: Dictionary = sim.get_snapshot()
			snap2["wall_time_unix"] = Time.get_unix_time_from_system()
			snap2["wall_time_monotonic_msec"] = Time.get_ticks_msec()
			_remember_decision_state(snap2)
			net.update_state(snap2)
			net.send_state(snap2)
		# check crash handled by signal
		_update_hud()
	# auto shake
	if shake_timer > 0:
		camera.position += Vector3(randf_range(-shake_intensity, shake_intensity), randf_range(-shake_intensity, shake_intensity), 0) * 0.1

func _handle_human_input(delta: float) -> void:
	if external_control:
		return
	# Brake is a held action, so it is polled; steering is handled by _input
	# so a press is acted on immediately instead of waiting for a poll.
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

func _process(delta: float) -> void:
	# pause toggle
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
	# net poll already in NetBridge._process
	# update conn label occasionally
	conn_label.text = "WS: " + ("connected" if net.connected_flag else "disconnected")
	if state == "menu":
		menu_panel.visible = true

func _key_just_pressed(key: int) -> bool:
	# fallback edge detection
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

func _reset_traffic_nodes() -> void:
	for c in traffic_root.get_children():
		c.queue_free()
	traffic_nodes.clear()

func _sync_traffic_nodes() -> void:
	# remove stale
	var current_ids := {}
	for v in sim.traffic:
		current_ids[v["id"]] = true
	for id in traffic_nodes.keys():
		if not current_ids.has(id):
			var n = traffic_nodes[id]
			if is_instance_valid(n):
				n.queue_free()
			traffic_nodes.erase(id)
	# add new
	for v in sim.traffic:
		if not traffic_nodes.has(v["id"]):
			var is_truck: bool = v["type"] == "truck"
			var color := Color(0.8, 0.2, 0.2) if not is_truck else Color(0.9, 0.6, 0.1)
			var node := _create_car(color, is_truck)
			traffic_root.add_child(node)
			traffic_nodes[v["id"]] = node
		var node = traffic_nodes[v["id"]]
		node.position = Vector3(sim.lane_center(int(v["lane"])), 0, v["z"])
	# update player
	player_node.position = Vector3(sim.player_x, 0, sim.player_z)
	player_node.rotation_degrees.y = 0
	player_node.rotation_degrees.z = clampf(-sim.player_velocity_x * 2.4, -24.0, 24.0)
	var brake_material := tail_light.material_override as StandardMaterial3D
	brake_material.albedo_color = Color(1.0, 0.08, 0.03) if sim.braking else Color(0.4, 0.03, 0.02)
	brake_material.emission = Color(1.0, 0.02, 0.01) if sim.braking else Color(0.25, 0.01, 0.0)
	# recycle road
	_recycle_road()
	_recycle_trees()
	_recycle_ground()

func _recycle_ground() -> void:
	for ground in ground_segments:
		while ground.position.z + GROUND_SEGMENT_LENGTH * 0.5 < sim.player_z - GROUND_RECYCLE_MARGIN:
			ground.position.z += GROUND_SEGMENT_LENGTH * GROUND_SEGMENT_COUNT

func _recycle_road() -> void:
	for seg in road_segments:
		while seg.position.z < sim.player_z - 30:
			seg.position.z += ROAD_SEGMENT_COUNT * ROAD_SEGMENT_LENGTH

## Puts the whole world back around the origin. A reset drops the player to
## z=0, so without this the road and scenery stay where the last episode
## ended and the player would start on bare ground.
func _reset_world_positions() -> void:
	for i in range(road_segments.size()):
		road_segments[i].position.z = i * ROAD_SEGMENT_LENGTH
	for i in range(ground_segments.size()):
		ground_segments[i].position.z = 120.0 + i * GROUND_SEGMENT_LENGTH
	for i in range(tree_nodes.size()):
		var side := -1 if i % 2 == 0 else 1
		tree_nodes[i].position = Vector3(side * rng_visual.randf_range(9.0, 16.0), 0, i * 16.0 - 60.0)
	for i in range(hill_nodes.size()):
		var hside := -1 if i % 2 == 0 else 1
		hill_nodes[i].position = Vector3(hside * rng_visual.randf_range(90.0, 220.0), 0, i * 260.0 - 200.0)
	for i in range(cloud_nodes.size()):
		cloud_nodes[i].position = Vector3(rng_visual.randf_range(-160.0, 160.0), rng_visual.randf_range(55.0, 95.0), i * 220.0 - 150.0)

func _recycle_trees() -> void:
	for t in tree_nodes:
		while t.position.z < sim.player_z - 60:
			t.position.z += 22 * 16.0
	for hill in hill_nodes:
		while hill.position.z < sim.player_z - 400:
			hill.position.z += 6 * 260.0
	for cloud in cloud_nodes:
		while cloud.position.z < sim.player_z - 300:
			cloud.position.z += 7 * 220.0

func _update_camera(force: bool) -> void:
	var target := Vector3(sim.player_x, 0, sim.player_z)
	var cam_pos := target + Vector3(0, 3.5, -5.6)
	if force:
		camera.position = cam_pos
	else:
		camera.position = camera.position.lerp(cam_pos, 0.12)
	camera.look_at(target + Vector3(0, 1.0, 10.5), Vector3.UP)
	# widen the view with speed: cheap, and it reads as acceleration
	var speed_ratio := clampf((sim.player_speed - sim.base_speed) / maxf(1.0, sim.max_speed - sim.base_speed), 0.0, 1.0)
	var target_fov := 60.0 + speed_ratio * 9.0
	camera.fov = lerpf(camera.fov, target_fov, 0.06 if not force else 1.0)

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
	auto_total_episodes = auto_seeds.size() * auto_episodes_per_seed
	if auto_max_episodes > 0:
		auto_total_episodes = min(auto_total_episodes, auto_max_episodes)
	# start first
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
