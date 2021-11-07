"""
Module for implementing the predictive analysis described in Pfeiffer and Foster (2013).
"""


import numpy as np


def get_path_by_time(start_time, end_time, pos_times, pos_xy):
    path_bool = (pos_times > start_time) & (pos_times < end_time)
    path = pos_xy[path_bool, :]
    return path


def get_path_by_dist_future(start_time, start_pos, pos_times, pos_xy, dist_thresh=50):
    path_start_ind = np.argwhere(pos_times > start_time)
    if len(path_start_ind) >= 1:
        path_start_ind = path_start_ind[0][0]
    elif len(path_start_ind) == 0:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    dist_to_start = np.sqrt(
        np.sum((pos_xy[path_start_ind:] - pos_xy[path_start_ind]) ** 2, axis=1)
    )
    path_end_ind = np.argwhere(dist_to_start > dist_thresh)
    if len(path_end_ind) >= 1:
        path_end_ind = path_start_ind + path_end_ind[0][0]
    elif len(path_end_ind) == 0:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    path = pos_xy[path_start_ind:path_end_ind, :]
    return path


def get_path_by_dist_past(end_time, end_pos, pos_times, pos_xy, dist_thresh=50):
    path_end_ind = np.argwhere(pos_times < end_time)
    if len(path_end_ind) >= 1:
        path_end_ind = path_end_ind[-1][0]
    elif len(path_end_ind) == 0:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    dist_to_end = np.sqrt(
        np.sum((pos_xy[:path_end_ind] - pos_xy[path_end_ind]) ** 2, axis=1)
    )
    path_start_ind = np.argwhere(dist_to_end > dist_thresh)
    if len(path_start_ind) >= 1:
        path_start_ind = path_start_ind[-1][0]
    elif len(path_start_ind) == 0:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    path = pos_xy[path_start_ind:path_end_ind, :]
    return path


def get_behavior_path(
    start_time, end_time, pos_times, pos_xy, dist_thresh=50, path_type=None
):
    path = get_path_by_time(start_time, end_time, pos_times, pos_xy)
    if len(path) != 0:
        dist_from_start = np.sqrt(np.sum((path - path[0]) ** 2, axis=1))
        if not np.any(dist_from_start > dist_thresh):
            if path_type == "future":
                start_pos = path[0]
                path = get_path_by_dist_future(
                    start_time, start_pos, pos_times, pos_xy, dist_thresh=dist_thresh
                )
            elif path_type == "past":
                start_pos = path[0]
                path = get_path_by_dist_past(
                    end_time, start_pos, pos_times, pos_xy, dist_thresh=dist_thresh
                )
            else:
                raise AttributeError("Invalid path type for selecting path by distance")
    if path_type == "past":
        path = np.flipud(path)
    return path


def get_circle(center_cm, radius_cm):
    points_x = np.linspace(-radius_cm, +radius_cm, 200)
    points_y = np.sqrt(radius_cm ** 2 - points_x ** 2)
    points_x = np.append(points_x, np.flip(points_x))
    points_y = np.append(points_y, -points_y)
    circle = np.array([points_x + center_cm[0], points_y + center_cm[1]]).T
    return circle


def get_point_on_circle(point1, point2, center, radius):
    if point1[0] == point2[0]:
        intersection1_x = point1[0]
        intersection2_x = point1[0]
        intersection1_y = center[1] + np.sqrt(
            radius ** 2 - (point1[0] - center[0]) ** 2
        )
        intersection2_y = center[1] - np.sqrt(
            radius ** 2 - (point1[0] - center[0]) ** 2
        )
    else:
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - m * point1[0]
        disc = radius ** 2 * (1 + m ** 2) - (center[1] - m * center[0] - b) ** 2
        intersection1_x = (center[0] + center[1] * m - b * m + np.sqrt(disc)) / (
            1 + m ** 2
        )
        intersection1_y = (
            b + center[0] * m + center[1] * m ** 2 + m * np.sqrt(disc)
        ) / (1 + m ** 2)
        intersection2_x = (center[0] + center[1] * m - b * m - np.sqrt(disc)) / (
            1 + m ** 2
        )
        intersection2_y = (
            b + center[0] * m + center[1] * m ** 2 - m * np.sqrt(disc)
        ) / (1 + m ** 2)
    dist1 = np.sum((intersection1_x - point1[0]) ** 2) + np.sum(
        (intersection1_y - point1[1]) ** 2
    )
    dist2 = np.sum((intersection2_x - point1[0]) ** 2) + np.sum(
        (intersection2_y - point1[1]) ** 2
    )
    if dist1 < dist2:
        return [intersection1_x, intersection1_y]
    elif dist1 > dist2:
        return [intersection2_x, intersection2_y]
    elif dist1 == dist2:
        return [intersection1_x, intersection1_y]
    else:
        print("some issue")


def get_intersection_array(center, traj, radius_array=np.arange(16, 60, 4)):
    crossing_array = np.empty((0, 2))
    distance_to_center = np.sqrt(np.sum((traj - center) ** 2, axis=1))
    for radius in radius_array:
        greater_than_radius = np.argwhere(np.diff(distance_to_center > radius))
        if len(greater_than_radius) >= 1:
            if (greater_than_radius[0] != 0) & (
                greater_than_radius[0] != (len(traj) - 1)
            ):
                crossing_ind = greater_than_radius[0]
                crossing = get_point_on_circle(
                    traj[crossing_ind][0], traj[crossing_ind + 1][0], center, radius
                )
            else:
                crossing = [np.nan, np.nan]
        else:
            crossing = [np.nan, np.nan]
        crossing_array = np.vstack((crossing_array, crossing))
    return crossing_array


def get_angular_distance(a, b, radius):
    # convert to polar coordinates
    a_theta = np.degrees(np.arctan(a[:, 1] / a[:, 0]))
    b_theta = np.degrees(np.arctan(b[:, 1] / b[:, 0]))
    for theta, x in zip([a_theta, b_theta], [a, b]):
        if ((x[:, 0] < 0) & (x[:, 1] > 0)).any():
            theta[(x[:, 0] < 0) & (x[:, 1] > 0)] = (
                theta[(x[:, 0] < 0) & (x[:, 1] > 0)] + 180
            )
        if ((x[:, 0] < 0) & (x[:, 1] < 0)).any():
            theta[(x[:, 0] < 0) & (x[:, 1] < 0)] = (
                theta[(x[:, 0] < 0) & (x[:, 1] < 0)] + 180
            )
        if ((x[:, 0] > 0) & (x[:, 1] < 0)).any():
            theta[(x[:, 0] > 0) & (x[:, 1] < 0)] = (
                theta[(x[:, 0] > 0) & (x[:, 1] < 0)] + 360
            )
    angular_dist = (b_theta - a_theta) % 360
    angular_dist[angular_dist > 180] = angular_dist[angular_dist > 180] - 360
    return angular_dist


def get_angular_dist_array(
    behavior_traj, replay_traj, radius_array=np.arange(16, 60, 4)
):
    center = behavior_traj[0]
    behavior_crossing_array = get_intersection_array(
        center, behavior_traj, radius_array=radius_array
    )
    replay_crossing_array = get_intersection_array(
        center, replay_traj, radius_array=radius_array
    )
    angular_distance_array = get_angular_distance(
        behavior_crossing_array - center, replay_crossing_array - center, radius_array
    )

    return angular_distance_array, behavior_crossing_array, replay_crossing_array
