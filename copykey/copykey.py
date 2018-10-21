import cv2
from copykey import video_to_frame, frame_to_boundary

def copykey(vid):
    frames = video_to_frame.get_frames(vid)

    key_images = []
    for frame in frames:
        success, key_image, _ = video_to_frame.process(frame)
        if success:
            key_images.append(key_image)
    result = video_to_frame.find_best(1, key_images)

    edges = frame_to_boundary.get_edges(result)
    boundary = frame_to_boundary.get_boundary_raytracing(edges, 10, 10)
    