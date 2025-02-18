import carla
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import csv

actor_list = []
IM_WIDTH = 640
IM_HEIGHT = 480

def get_vehicle_speed(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed_mps * 3.6  # Convert m/s to km/h

def detect_curved_lanes(image):
    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(img, mask)
        return masked
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height, width = image.shape[:2]
    roi_vertices = np.array([[(0, height), 
                              (width * 0.45, height * 0.6),
                              (width * 0.55, height * 0.6), 
                              (width, height)]], dtype=np.int32)
    
    roi = region_of_interest(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    left_lines, right_lines = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    def fit_poly(lines):
        if not lines:
            return None
        points_x, points_y = [], []
        for line in lines:
            x1, y1, x2, y2 = line
            points_x.extend([x1, x2])
            points_y.extend([y1, y2])
        if len(points_x) >= 2:
            curve = np.polyfit(points_y, points_x, 2)
            if abs(curve[0]) <= 0.02:
                return curve
        return None

    left_curve, right_curve = fit_poly(left_lines), fit_poly(right_lines)
    result = image.copy()
    plot_y = np.linspace(height * 0.6, height, num=20)

    if left_curve is not None:
        plot_x = left_curve[0] * plot_y**2 + left_curve[1] * plot_y + left_curve[2]
        pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], dtype=np.int32)
        cv2.polylines(result, pts, False, (255, 0, 0), thickness=4)
    if right_curve is not None:
        plot_x = right_curve[0] * plot_y**2 + right_curve[1] * plot_y + right_curve[2]
        pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], dtype=np.int32)
        cv2.polylines(result, pts, False, (0, 0, 255), thickness=4)

    return result, left_curve, right_curve

def calculate_lane_distances(left_curve, right_curve, height):
    center_x = IM_WIDTH // 2
    y_eval = height - 1
    
    left_x = (left_curve[0] * y_eval**2 + left_curve[1] * y_eval + left_curve[2]) if left_curve is not None else None
    right_x = (right_curve[0] * y_eval**2 + right_curve[1] * y_eval + right_curve[2]) if right_curve is not None else None
    
    left_distance = center_x - left_x if left_x is not None else None
    right_distance = right_x - center_x if right_x is not None else None

    return left_distance, right_distance

def process_img(image, vehicle):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    original = i2[:,:,:3].astype(np.uint8)

    lane_image, left_curve, right_curve = detect_curved_lanes(original)
    left_distance, right_distance = calculate_lane_distances(left_curve, right_curve, IM_HEIGHT)
    
    # Determine steering angle based on lane distances
    # Left Lane Distance
    if left_distance is not None and 200 <= left_distance <= 230:
        steering_angle = 0
    elif left_distance is not None and left_distance > 230:
        steering_angle = -0.1
    elif left_distance is not None and left_distance < 200 and left_distance >= 160:
        steering_angle = 0.1
    elif left_distance is not None and left_distance < 160 and left_distance >= 100:
        steering_angle = 0.25
    # Right Lane Distance
    elif right_distance is not None and 200 <= right_distance <= 230:
        steering_angle = 0
    elif right_distance is not None and right_distance > 230:
        steering_angle = 0.1
    elif right_distance is not None and right_distance < 200 and right_distance >= 160:
        steering_angle = -0.1
    elif right_distance is not None and right_distance < 160 and right_distance >= 100:
        steering_angle = -0.25
    # Other Case
    else:
        steering_angle = 0

    # Map steering angle to class labels
    class_label = {
        0: "straight",
        0.1: "light right",
        0.25: "right",
        -0.1: "light left",
        -0.25: "left"
    }.get(steering_angle, "unknown")

    # Store data
    lane_data = {
        "Left Distance": left_distance,
        "Right Distance": right_distance,
        "Steering Angle": steering_angle,
        "Class": class_label
    }
    print(lane_data)

    # Save data to dataset.csv
    with open("dataset.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([left_distance, right_distance, steering_angle, class_label])

    # Display lane distances and control information
    if left_distance is not None and left_distance > 0:
        cv2.putText(lane_image, f"Left Dist: {left_distance:.2f}", (10, IM_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if right_distance is not None and right_distance > 0:
        cv2.putText(lane_image, f"Right Dist: {right_distance:.2f}", (IM_WIDTH - 250, IM_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Throttle and brake logic based on curvature
    throttle = 0.2 if abs(steering_angle) > 0.25 else 0.5
    if abs(steering_angle) >= 0.25: 
        brake = 0.3
    elif abs(steering_angle) >= 0.1:
        brake = 0.1
    else: brake = 0.0
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steering_angle))

    # Draw steering indicator
    cv2.line(lane_image, 
             (IM_WIDTH//2, IM_HEIGHT),
             (int(IM_WIDTH//2 + steering_angle * 100), IM_HEIGHT-50),
             (255, 0, 0), 3)

    # Display speed, speed limit and control data
    speed = get_vehicle_speed(vehicle)
    cv2.putText(lane_image, f"Speed: {speed:.2f} km/h", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    speed_limit = vehicle.get_speed_limit()
    cv2.putText(lane_image, f"Limit: {speed_limit} km/h", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Lane Detection with Curves", lane_image)
    cv2.waitKey(1)
    # Following the vehicle function
    vehicle_follower(vehicle=vehicle)
    return original / 255.0

def vehicle_follower(vehicle):
    def calculate_sides(hypotenuse, angle):
        # Convert the angle to radians
        angle_radians = math.radians(angle)

        # Calculate the opposite side using the sine function
        opposite_side = hypotenuse * math.sin(angle_radians)

        # Calculate the adjacent side using the cosine function
        adjacent_side = hypotenuse * math.cos(angle_radians)

        return opposite_side, adjacent_side
    # follow the car
    # here we subtract the delta x and y to be behind 
    metres_distance = 5
    vehicle_transform = vehicle.get_transform()
    
    # Spectator will follow the car
    y,x = calculate_sides(metres_distance, vehicle_transform.rotation.yaw )

    spectator_pos = carla.Transform(vehicle_transform.location + carla.Location(x=-x,y=-y,z=5 ),
                                            carla.Rotation( yaw = vehicle_transform.rotation.yaw,pitch = -25))
    spectator.set_transform(spectator_pos)        
    
try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    client.load_world('Town04_Opt')

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[20]
    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{    IM_HEIGHT}")
    cam_bp.set_attribute("fov", "100")
    spawn_point = carla.Transform(carla.Location(x=1, z=2))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)

    # get spectator
    spectator = world.get_spectator()
    spectator_pos = carla.Transform(spawn_point.location + carla.Location(x=20,y=10,z=4),
                                    carla.Rotation(yaw = spawn_point.rotation.yaw -155))
    spectator.set_transform(spectator_pos)
    actor_list.append(spectator)

    sensor.listen(lambda data: process_img(data, vehicle))
    time.sleep(600)

finally:
    for actor in actor_list:
        actor.destroy()
    cv2.destroyAllWindows()
