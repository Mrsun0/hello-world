import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Point = Tuple[float, float]


@dataclass
class CirclePoint:
    center: Point
    radius: float


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def is_internal_circle(center: Point, radius: float, polygon: Sequence[Point]) -> bool:
    if not point_in_polygon(center, polygon):
        return False

    n = len(polygon)
    min_dist = float("inf")
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        min_dist = min(min_dist, point_to_segment_distance(center, a, b))
    return min_dist > radius


def classify_circles(
    polygon: Sequence[Point], circle_centers: Iterable[Point], radius: float
) -> Tuple[List[CirclePoint], List[CirclePoint]]:
    internal: List[CirclePoint] = []
    boundary_related: List[CirclePoint] = []

    for c in circle_centers:
        circle = CirclePoint(center=c, radius=radius)
        if is_internal_circle(c, radius, polygon):
            internal.append(circle)
        else:
            boundary_related.append(circle)

    return internal, boundary_related


def k_nearest_indices(points: Sequence[Point], idx: int, k: int) -> List[int]:
    dists: List[Tuple[float, int]] = []
    x0, y0 = points[idx]
    for j, (x, y) in enumerate(points):
        if j == idx:
            continue
        dists.append((math.hypot(x - x0, y - y0), j))
    dists.sort(key=lambda item: item[0])
    return [j for _, j in dists[:k]]


def build_mutual_knn_graph(points: Sequence[Point], k: int = 6) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = {i: set() for i in range(len(points))}
    knn = {i: set(k_nearest_indices(points, i, k)) for i in range(len(points))}
    for i in range(len(points)):
        for j in knn[i]:
            if i in knn[j]:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def find_core_index(points: Sequence[Point]) -> int:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return min(range(len(points)), key=lambda i: math.hypot(points[i][0] - cx, points[i][1] - cy))


def regularize_global_hexagonal(internal: Sequence[CirclePoint], radius: float) -> Tuple[List[CirclePoint], str]:
    """Heuristic global lattice regularization.

    Steps:
    1) pick core point near centroid
    2) require exactly 6 mutual-KNN neighbors for first ring
    3) place first ring onto perfect hexagon around core, preserving average scale/rotation
    4) BFS outward, snapping each new node from a fixed parent along nearest 60-degree direction
    """
    if len(internal) < 7:
        return list(internal), "Not enough internal circles (<7), skip regularization."

    points = [c.center for c in internal]
    graph = build_mutual_knn_graph(points, k=min(6, len(points) - 1))
    core = find_core_index(points)
    ring = sorted(graph[core], key=lambda j: math.atan2(points[j][1] - points[core][1], points[j][0] - points[core][0]))
    if len(ring) != 6:
        return list(internal), f"Core neighbor count is {len(ring)} (not 6), skip regularization."

    lattice = [tuple(p) for p in points]
    px, py = points[core]

    mean_angle = sum(math.atan2(points[j][1] - py, points[j][0] - px) for j in ring) / 6.0
    edge_len = math.sqrt(3.0) * radius

    for m, j in enumerate(ring):
        ang = mean_angle + m * (math.pi / 3.0)
        lattice[j] = (px + edge_len * math.cos(ang), py + edge_len * math.sin(ang))

    fixed = {core, *ring}
    queue = list(ring)

    while queue:
        parent = queue.pop(0)
        parent_pos = lattice[parent]
        parent_neighbors = sorted(graph[parent], key=lambda j: math.atan2(points[j][1] - points[parent][1], points[j][0] - points[parent][0]))

        for nb in parent_neighbors:
            if nb in fixed:
                continue
            # keep approximate original direction, snap to nearest 60-degree orientation
            raw_ang = math.atan2(points[nb][1] - points[parent][1], points[nb][0] - points[parent][0])
            rel = (raw_ang - mean_angle) / (math.pi / 3.0)
            snapped = round(rel) * (math.pi / 3.0) + mean_angle
            lattice[nb] = (
                parent_pos[0] + edge_len * math.cos(snapped),
                parent_pos[1] + edge_len * math.sin(snapped),
            )
            fixed.add(nb)
            queue.append(nb)

    out = [CirclePoint(center=lattice[i], radius=radius) for i in range(len(internal))]
    return out, f"Regularization applied. core={core}, first-ring=6, fixed={len(fixed)}/{len(internal)}"


def export_svg(
    polygon: Sequence[Point],
    internal: Sequence[CirclePoint],
    boundary_related: Sequence[CirclePoint],
    output_path: str,
    title: str,
) -> None:
    xs = [p[0] for p in polygon] + [c.center[0] for c in internal] + [c.center[0] for c in boundary_related]
    ys = [p[1] for p in polygon] + [c.center[1] for c in internal] + [c.center[1] for c in boundary_related]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad = 1.5
    width = max_x - min_x + 2 * pad
    height = max_y - min_y + 2 * pad

    def map_xy(p: Point) -> Point:
        x, y = p
        return x - min_x + pad, height - (y - min_y + pad)

    polygon_points = " ".join(f"{map_xy(p)[0]:.2f},{map_xy(p)[1]:.2f}" for p in polygon)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="700" viewBox="0 0 {width:.2f} {height:.2f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<polygon points="{polygon_points}" fill="none" stroke="black" stroke-width="0.08"/>',
        f'<text x="0.3" y="0.6" font-size="0.45">{title}</text>',
    ]

    for c in internal:
        cx, cy = map_xy(c.center)
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{c.radius:.2f}" fill="none" stroke="green" stroke-width="0.07"/>')
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.08" fill="green"/>')

    for c in boundary_related:
        cx, cy = map_xy(c.center)
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{c.radius:.2f}" fill="none" stroke="red" stroke-width="0.07"/>')
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.08" fill="red"/>')

    svg.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def demo() -> None:
    polygons = [
    (2165, 250), (1732, 250), (1732, 1000), (1299, 1000), (1299, 1750),
    (-1299, 1750), (-1299, 1000), (-1732, 1000), (-1732, 250),
    (-2165, 250), (-2165, -250), (-1732, -250), (-1732, -1000),
    (-1299, -1000), (-1299, -1750), (1299, -1750), (1299, -1000),
    (1732, -1000), (1732,-250, (2165, -250), (2165, 250),
    ]
    centers = [
        (-855.0810478152164, -1487.6675363799322),
        (930.8523762994432, 0.6223726126537872),
        (1315.5959662025323, -733.48691141416448),
        (404.9264063740799, -724.1967849115928),
        (858.8213557215867, 1515.9012439066956),
        (864.729983721718, -1466.4366095302666),
        (-83.92259661057244, 1424.5335566067911),
        (-889.8385114725363, 1494.6899472463447),
        (425.15731469127104, 702.4281142075728),
        (1775.5764068197717, 59.94176857236738),
        (-1748.6262601985782, 4.39413489961058),
        (1309.0603202155132, 805.2863827273463),
        (-1259.7557894356375, 795.4977983603724),
        (-477.35250723871945, -791.8119955145997),
        (-35.648880254673325, -11.81661036789848113),
        (844.1775872261513, 30.886557426161515),
        (-420.23137528303454, 729.28831459697875),
        (-1385.91538135036564, -793.90846884682371),
        (11.272958043798077, -1574.81368638121108),
    ]
    radius = 500.0

    internal, boundary_related = classify_circles(polygon, centers, radius)
    print(f"Total circles: {len(centers)}")
    print(f"Internal circles: {len(internal)}")
    print(f"Boundary circles: {len(boundary_related)}")

    export_svg(polygon, internal, boundary_related, "circle_classification_demo.svg", "Step1: Green=internal circles, Red=boundary circles")

    regularized_internal, msg = regularize_global_hexagonal(internal, radius)
    print(msg)
    export_svg(
        polygon,
        regularized_internal,
        boundary_related,
        "circle_regularized_global_demo.svg",
        "Step2 global lattice regularization: green moved to hexagonal lattice",
    )
    print("Saved figures: circle_classification_demo.svg, circle_regularized_global_demo.svg")


if __name__ == "__main__":
    demo()
