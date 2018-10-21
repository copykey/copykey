import cv2
import numpy as np
from string import Template

def align(boundary):
    """
    Rotates the boundary based on the baseline. Returns the rotated boundary and the new
    y-value of the baseline.
    """
    tip_index = np.argmax(boundary[:, 0])
    tip = boundary[tip_index]

    boundary_bottom = boundary[tip_index:]

    # between 15 and 5 mm back from tip
    baseline_start = tip[0] - 15
    baseline_end = tip[0] - 5
    baseline = boundary_bottom[np.logical_and(boundary_bottom[:, 0] >= baseline_start, boundary_bottom[:, 0] <= baseline_end)]

    # shift baseline to origin
    baseline_origin = baseline - baseline[-1]
    slope = np.linalg.lstsq(baseline_origin[:, 0].reshape(-1, 1), baseline_origin[:, 1], rcond=None)[0]
    angle = -1 * np.asscalar(np.arctan(slope))
    rotation_matrix = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    boundary_rotated = rotation_matrix.dot(boundary.T).T
    baseline_rotated = rotation_matrix.dot(baseline.T).T
    return boundary_rotated, np.mean(baseline_rotated[:, 1])
    # return baseline[0], baseline[-1]


def boundary_to_scad(boundary, baseline, keytype):
    template = None
    with open("template.scad", "r") as f:
        template = Template(f.read())
    return template.substitute(keyway_name=keytype, baseline=baseline, keypoints=str(boundary.tolist()))
