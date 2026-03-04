import carla
import random

def setup_carla_world():
    print("Connecting to CARLA on localhost:2000...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'ego_vehicle':
            actor.destroy()

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('model3')[0]
    vehicle_bp.set_attribute('role_name', 'ego_vehicle')
    
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    ego_vehicle = None
    for sp in spawn_points:
        ego_vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if ego_vehicle is not None:
            break
            
    if ego_vehicle is None:
        raise RuntimeError("Could not find a free spawn point. Restart CARLA server.")

    # --- THE FIX: TRUE DASHCAM MOUNT ---
    # x=1.5 (forward on the hood), z=1.3 (eye level), pitch=-2 (looking slightly down at the road)
    cam_trans = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.3), carla.Rotation(pitch=-2.0))
    
    # Force 640x480 resolution for consistent matrix math
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '640')
    rgb_bp.set_attribute('image_size_y', '480')
    rgb_cam = world.spawn_actor(rgb_bp, cam_trans, attach_to=ego_vehicle)

    depth_bp = bp_lib.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', '640')
    depth_bp.set_attribute('image_size_y', '480')
    depth_cam = world.spawn_actor(depth_bp, cam_trans, attach_to=ego_vehicle)

    return world, ego_vehicle, rgb_cam, depth_cam, [ego_vehicle, rgb_cam, depth_cam]

def cleanup(actor_list):
    for actor in actor_list:
        if actor is not None and hasattr(actor, 'destroy'):
            actor.destroy()