import cv2
import numpy as np


def get_frames(vid):
    if not vid.isOpened():
        return False
    frames = []
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            frames.append(frame)
        else:
            break
    vid.release()
    return frames


# returns the intersection of two lists which each contain 2 items
def list_intersection_len_2(list1, list2):
    ret = []
    if list1[0] in list2:
        ret.append(list1[0])
    if list1[1] in list2:
        ret.append(list1[1])
    return ret


# takes in list of points and returns them in order (bottom_left, bottom_right, top_left, top_right)
# honestly this is a lil disgusting and maybe it should be changed later but it works so w/e
def normalize_points(contour_orig):
    contour = []
    for i in range(0, len(contour_orig)):
        contour.append((contour_orig[i][0][0], contour_orig[i][0][1]))
    assert (len(contour) == 4)
    xs = [a[0] for a in contour]
    xs_s = sorted(xs)
    ys = [a[1] for a in contour]
    ys_s = sorted(ys)
    avg_x = (xs_s[1] + xs_s[2]) / 2
    avg_y = (ys_s[1] + ys_s[2]) / 2
    left = []
    top = []
    right = []
    bottom = []
    for point in contour:
        if len(left) < 2 and point[0] <= avg_x:
            left.append(point)
        elif len(right) < 2 and point[0] >= avg_x:
            right.append(point)
        if len(top) < 2 and point[1] <= avg_y:
            top.append(point)
        elif len(bottom) < 2 and point[1] >= avg_y:
            bottom.append(point)
    assert (len(left) == 2)
    assert (len(right) == 2)
    assert (len(top) == 2)
    assert (len(bottom) == 2)
    top_lefts = list_intersection_len_2(top, left)
    top_rights = list_intersection_len_2(top, right)
    bottom_lefts = list_intersection_len_2(bottom, left)
    bottom_rights = list_intersection_len_2(bottom, right)
    assert (len(top_lefts) == 1)
    assert (len(top_rights) == 1)
    assert (len(bottom_lefts) == 1)
    assert (len(bottom_rights) == 1)
    return [bottom_lefts[0], bottom_rights[0], top_lefts[0], top_rights[0]]


# loads in a color image. returns a tuple containing the image with the key and then the image with stuff drawn
def process(img_c):
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    img_res = img_c.copy()

    ret, thresh1 = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        percentarea = area / (height * width)
        if percentarea < 0.05 or percentarea > 0.5:
            continue
        cv2.rectangle(img_c, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.drawContours(img_c, [cnt], 0, (0, 255, 255), 5)
        # rotated rect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_c, [box], 0, (0, 0, 255), 1)
        # convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img_c, [hull], 0, (255, 0, 255), 2)
        # approximate polygon
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_c, [approx], 0, (100, 100, 255), 2)
        approx_area = cv2.contourArea(approx)
        area_diff = area / approx_area if area < approx_area else approx_area / area
        if area_diff < 0.95:
            continue
        # draw text
        cv2.putText(img_c,
                    str(area_diff) + "---" + str(len(cnt)) + "---" + str(len(approx)) + "----" + str(percentarea),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        # warp
        if len(approx) != 4:
            continue
        pts1 = np.float32(normalize_points(approx))
        new_width = h * 2  # calculate new based on knowledge that width of rectangle is twice its height
        new_height = h
        pts2 = np.float32([[0, new_height], [new_width, new_height], [0, 0], [new_width, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        crop_px = new_width // 30  # magic number. crop off these many pixels per edege
        dst = cv2.warpPerspective(img_res, M, (new_width, new_height))[crop_px:new_height - crop_px,
              crop_px:new_width - crop_px]
        return (True, dst, img_c)
    return (False, None, img_res)


def find_best(num_desired, images):
    laplacian_variances = []
    for image in images:
        height, width, _ = image.shape
        canny = cv2.Canny(image, 60, 135)
        laplacian_variances.append([image, cv2.Laplacian(canny, cv2.CV_64F).var() * width])
    laplacian_variances.sort(key=lambda x: -x[1])
    return [i[0] for i in laplacian_variances[0:num_desired]]
