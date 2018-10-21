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

def get_pixel_conversion(width, height):
    def conversion(point):
        scale = WIDTH_MM / width
        x = point[0] / scale
        y = height - (point[1] / scale) - 1
        return np.array([int(x), int(y)])
    return conversion

def get_preview_from_boundary(width, height, boundary, img=None, baseline=None):
    if img is None:
        img = np.full((height, width, 3), [0, 0, 0], dtype=np.uint8)
    for i in range(-1, len(boundary) - 1):
        cv2.line(img, tuple(boundary[i]), tuple(boundary[i + 1]), (0, 255, 0), 1)

    if baseline is not None:
        img[baseline] = np.bitwise_or(img[baseline], [0, 0, 255])
    return img

def copykey(input, output, keytype, cool_video_output=None):
    frames = video_to_frame.get_frames(cv2.VideoCapture(input))

    key_images = []
    cool_images = []
    for frame in frames:
        success, key_image, cool_image = video_to_frame.process(frame)
        if success:
            key_images.append(key_image)
            cool_images.append(cool_image)

    result = video_to_frame.find_best(1, key_images)[0]
    height, width, _ = result.shape

    if cool_video_output is not None:
        os.remove(cool_video_output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cool_video_output + ".tmp.mp4", fourcc, 20.0, (cool_images[0].shape[1], cool_images[0].shape[0]))
        for image in cool_images:
            out.write(image)
        out.release()
        os.system("ffmpeg -i {} {}".format(cool_video_output + ".tmp.mp4", cool_video_output))
        os.remove(cool_video_output + ".tmp.mp4")

    result = cv2.bilateralFilter(result, 10, 25, 25)


    edges = frame_to_boundary.get_edges(result, low=65, high=165)
    boundary = frame_to_boundary.get_boundary_raytracing(edges, filter_threshold=int(0.01 * width / WIDTH_MM), filter_window_size=int(0.01 * width / WIDTH_MM))

    boundary_transformed = np.apply_along_axis(get_mm_conversion(width, height), axis=1, arr=boundary)
    boundary_rotated, baseline = boundary_to_stl.align(boundary_transformed)


    scadstring = boundary_to_stl.boundary_to_scad(boundary_rotated, baseline, keytype)

    with open(output, "w+") as f:
        f.write(scadstring)

    to_pixel_conversion = get_pixel_conversion(width, height)
    preview1 = get_preview_from_boundary(width, height, boundary, img=result)
    preview2 = get_preview_from_boundary(width, height, np.apply_along_axis(to_pixel_conversion, axis=1, arr=boundary_rotated), baseline=to_pixel_conversion([0, baseline])[1])
    return preview1, preview2
