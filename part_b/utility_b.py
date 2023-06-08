import numpy as np

def ransac_plane_fit(points, n_iterations, threshold_distance):
    best_plane = None
    best_inliers = []
    max_inliers_count = 0

    for _ in range(n_iterations):
        # Randomly select three points
        indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[indices]

        # Check if the points are non-collinear
        # collinear points are the points which are on the same line
        if not are_points_non_collinear(sample_points):
            continue

        # Fit a plane to the sampled points
        plane = create_plane(sample_points)

        # Compute the distances from all points to the plane
        distances = compute_distances_for_plane(points, plane)

        # Count inliers (points within threshold distance)
        inliers = np.where(distances < threshold_distance)[0]
        inliers_count = len(inliers)

        # Check if current model is the best so far
        if inliers_count > max_inliers_count:
            max_inliers_count = inliers_count
            best_inliers = inliers
            best_plane = plane

    return best_plane, best_inliers


def compute_distances_for_plane(points, plane):
    normal = plane[:3]
    d = plane[3]
    distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
    return distances

def create_plane(points):
    vector1 = points[1] - points[0]
    vector2 = points[2] - points[0]
    normal = np.cross(vector1, vector2)
    normal /= np.linalg.norm(normal)
    d = -np.dot(normal, points[0])
    return np.concatenate((normal, [d]))

def are_points_non_collinear(points):
    # Calculate slopes between pairs of points
    try :
        slope1 = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
        slope2 = (points[2, 1] - points[1, 1]) / (points[2, 0] - points[1, 0])
        slope3 = (points[2, 1] - points[0, 1]) / (points[2, 0] - points[0, 0])
    except ZeroDivisionError:
        return False

    # Check if all slopes are distinct
    return slope1 != slope2 and slope1 != slope3 and slope2 != slope3

def compute_distances_for_line(point1, point2, points):
    # Compute the direction vector of the line
    direction_vector = point2 - point1

    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)

    # Compute the vector from point1 to each point in the array
    vector_to_points = points - point1

    # Compute the projection of vector_to_points onto the direction vector
    projection = np.dot(vector_to_points, direction_vector)

    # Compute the minimum distance for each point
    distances = np.linalg.norm(vector_to_points - np.outer(projection, direction_vector), axis=1)

    return distances

def ransac_line_fit(points, n_iterations, threshold_distance):
    best_inliers = []
    max_inliers_count = 0

    for _ in range(n_iterations):
        # randomly select two points
        indices = np.random.choice(len(points), 2, replace=False)
        sample_points = points[indices]

        # if the points are the same continue
        if np.all(sample_points[0] == sample_points[1]):
            continue

        # compute the distances from all points to the line
        distances = compute_distances_for_line(sample_points[0],sample_points[1], points)

        # count inliers (points within threshold distance)
        inliers = np.where(distances < threshold_distance)[0]
        inliers_count = len(inliers)

        # check if current model is the best so far
        if inliers_count > max_inliers_count:
            max_inliers_count = inliers_count
            best_inliers = inliers
        
    return best_inliers
