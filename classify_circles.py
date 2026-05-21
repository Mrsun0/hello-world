import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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


def export_svg(
    polygon: Sequence[Point],
    internal: Sequence[CirclePoint],
    boundary_related: Sequence[CirclePoint],
    output_path: str = "circle_classification_demo.svg",
) -> None:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
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
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="700" viewBox="0 0 {width:.2f} {height:.2f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<polygon points="{polygon_points}" fill="none" stroke="black" stroke-width="0.08"/>',
        '<text x="0.3" y="0.6" font-size="0.45">Green=internal circles, Red=boundary circles</text>',
    ]

    for c in internal:
        cx, cy = map_xy(c.center)
        svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{c.radius:.2f}" fill="none" stroke="green" stroke-width="0.07"/>'
        )
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.08" fill="green"/>')

    for c in boundary_related:
        cx, cy = map_xy(c.center)
        svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{c.radius:.2f}" fill="none" stroke="red" stroke-width="0.07"/>'
        )
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.08" fill="red"/>')

    svg.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def demo() -> None:
    polygon = [(0, 0), (10, 0), (12, 4), (9, 8), (4, 10), (0, 6)]
    centers = [
        (2.0, 2.0), (4.5, 2.0), (7.0, 2.2), (9.5, 2.3), (2.0, 5.0),
        (4.8, 5.0), (7.2, 5.0), (9.6, 5.8), (3.0, 8.0), (6.0, 8.0), (8.8, 7.3),
    ]
    radius = 1.0

    internal, boundary_related = classify_circles(polygon, centers, radius)

    print(f"Total circles: {len(centers)}")
    print(f"Internal circles: {len(internal)}")
    print(f"Boundary circles: {len(boundary_related)}")

    export_svg(polygon, internal, boundary_related)
    print("Saved figure: circle_classification_demo.svg")


if __name__ == "__main__":
    demo()
