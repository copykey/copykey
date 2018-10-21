import cv2
import numpy as np
from copykey import video_to_frame, frame_to_boundary, boundary_to_stl
import os

# width of image in mm
WIDTH_MM = 94.8266666667

def get_mm_conversion(width, height):
    """
    Creates a method to converts a point in pixel coordinates (with (0, 0) at top left)
    to a point in real coordinates (with (0, 0) at bottom left) in millimeters
    """
    def conversion(point):
        x = point[0]
        y = height - point[1] - 1
        scale = WIDTH_MM / width
        return np.array([x * scale, y * scale])
    return conversion

def convert_back(point, width, height):
    scale = WIDTH_MM / width
    x = point[0] / scale
    y = height - (point[1] / scale) - 1
    return np.array([int(x), int(y)])


def copykey(input, output, keytype):
    frames = video_to_frame.get_frames(cv2.VideoCapture(input))

    key_images = []
    for frame in frames:
        success, key_image, _ = video_to_frame.process(frame)
        if success:
            key_images.append(key_image)
    result = video_to_frame.find_best(1, key_images)[0]

    edges = frame_to_boundary.get_edges(result)
    boundary = frame_to_boundary.get_boundary_raytracing(edges, int(0.1 * edges.shape[1] / WIDTH_MM), int(0.1 * edges.shape[1] / WIDTH_MM))

    boundary_transformed = np.apply_along_axis(get_mm_conversion(result.shape[1], result.shape[0]), axis=1, arr=boundary)
    scadstring = boundary_to_stl.boundary_to_scad(boundary_transformed, keytype)

    f = open(output, "w+")
    f.write(scadstring)

    os.remove(input)