"""
Layered rendering system for Smart Warehouse Simulator.

Architecture:
  draw_frame()
   ├── build_background()   → brick wall + floor + rack slots + markers
   ├── build_objects()      → boxes + robot sprite
   ├── build_overlay()      → path + events
   └── compose()            → final blended image
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image

# ── TILE CONFIG ──
TILE = 32  # pixels per grid cell

# ── ASSET LOADING (cached) ──
@st.cache_data
def load_assets():
    """Load sprite images from disk (cached by Streamlit)."""
    _ensure_placeholder_images()
    base = os.path.dirname(__file__)
    brick = mpimg.imread(os.path.join(base, "brick.jpeg"))
    robot = mpimg.imread(os.path.join(base, "robot.png"))
    box = mpimg.imread(os.path.join(base, "box.png"))
    return brick, robot, box


def _ensure_placeholder_images():
    """Create simple placeholder sprites if image files don't exist yet."""
    base = os.path.dirname(__file__)

    # ── brick.jpeg (RGB) ──
    brick_path = os.path.join(base, "brick.jpeg")
    if not os.path.exists(brick_path):
        img = Image.new("RGB", (TILE, TILE), (180, 80, 60))  # brick red-brown
        px = img.load()
        for i in range(TILE):
            for j in range(TILE):
                # mortar lines every 8 pixels
                if i % 8 < 2 or j % 8 < 2:
                    px[i, j] = (210, 205, 195)
                # brick highlight
                elif (i // 8 + j // 8) % 2 == 0:
                    px[i, j] = (190, 90, 65)
        img.save(brick_path)

    # ── robot.png (RGBA) ──
    robot_path = os.path.join(base, "robot.png")
    if not os.path.exists(robot_path):
        img = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
        px = img.load()
        for i in range(TILE):
            for j in range(TILE):
                # body (rounded rectangle)
                if 4 <= i <= 27 and 4 <= j <= 27:
                    # rounded corners
                    if (i - 4) ** 2 + (j - 4) ** 2 > 36 and \
                       (i - 4) ** 2 + (j - 27) ** 2 > 36 and \
                       (i - 27) ** 2 + (j - 4) ** 2 > 36 and \
                       (i - 27) ** 2 + (j - 27) ** 2 > 36:
                        px[i, j] = (70, 130, 210, 255)  # blue body
                    else:
                        px[i, j] = (70, 130, 210, 255)
                # eyes (white with black pupil)
                if 8 <= i <= 12 and 8 <= j <= 12:
                    px[i, j] = (255, 255, 255, 255)
                if 9 <= i <= 11 and 9 <= j <= 11:
                    px[i, j] = (0, 0, 0, 255)
                if 8 <= i <= 12 and 20 <= j <= 24:
                    px[i, j] = (255, 255, 255, 255)
                if 9 <= i <= 11 and 21 <= j <= 23:
                    px[i, j] = (0, 0, 0, 255)
                # antenna
                if 14 <= i <= 16 and 1 <= j <= 3:
                    px[i, j] = (200, 50, 50, 255)
                if 15 == i and 0 <= j <= 1:
                    px[i, j] = (255, 200, 50, 255)
        img.save(robot_path)

    # ── box.png (RGBA) ──
    box_path = os.path.join(base, "box.png")
    if not os.path.exists(box_path):
        img = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
        px = img.load()
        for i in range(TILE):
            for j in range(TILE):
                # box body
                if 2 <= i <= 29 and 2 <= j <= 29:
                    px[i, j] = (190, 140, 70, 255)  # cardboard brown
                # border
                if 2 <= i <= 29 and (j == 2 or j == 29):
                    px[i, j] = (140, 100, 50, 255)
                if (i == 2 or i == 29) and 2 <= j <= 29:
                    px[i, j] = (140, 100, 50, 255)
                # cross tape pattern
                if 14 <= i <= 18 or 14 <= j <= 18:
                    px[i, j] = (210, 180, 120, 255)
        img.save(box_path)


# Load assets at module level (cached)
brick_img, robot_img, box_img = load_assets()


# ── RESIZE HELPER ──
def resize(img, size):
    """
    Resize an image to the given (width, height) in pixels.
    Handles both RGB and RGBA images in float [0,1] range.
    """
    # Handle grayscale (2D) → RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    # Convert float [0,1] → uint8 [0,255]
    if img.dtype == np.float32 or img.dtype == np.float64:
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        arr = img.astype(np.uint8)
    # Resize with LANCZOS for quality
    pil_img = Image.fromarray(arr)
    resized = pil_img.resize(size, Image.LANCZOS)
    return np.array(resized).astype(np.float32) / 255.0


# ── ROAD TEXTURE GENERATOR ──
@st.cache_data
def _generate_road_tile():
    """Generate a procedural road/asphalt tile (TILExTILE, RGB) with realistic texture."""
    tile = np.ones((TILE, TILE, 3))
    rng = np.random.RandomState(42)  # fixed seed for consistent texture
    base = 0.25 + rng.random((TILE, TILE)) * 0.08  # dark grey base with noise
    tile[:, :, 0] = base * 0.22 / 0.29  # adjust R channel
    tile[:, :, 1] = base * 0.22 / 0.29  # adjust G channel
    tile[:, :, 2] = base * 0.24 / 0.29  # adjust B channel
    # Add some darker speckles for pebble texture
    speckle = rng.random((TILE, TILE)) > 0.92
    tile[speckle] *= 0.7
    # Add subtle lighter speckles
    light = rng.random((TILE, TILE)) > 0.96
    tile[light] = tile[light] * 1.3 + 0.05
    # Subtle horizontal lane marking hint (faded)
    mid = TILE // 2
    for i in range(max(0, mid - 2), min(TILE, mid + 2)):
        for j in range(0, TILE, 4):
            tile[i, j:j+2] = tile[i, j:j+2] * 0.6 + 0.35  # faded lane dots
    return np.clip(tile, 0.0, 1.0)


# ── LAYER 1: BACKGROUND ──
def build_background(env):
    """
    Build the static background layer:
    - Walkable tiles: road texture
    - Walls (1): brick texture
    - Rack slots (2): orange (empty) / green (filled)
    - Start: cyan
    - Finish: magenta
    """
    h, w = env.size, env.size
    road_tile = _generate_road_tile()
    bg = np.tile(road_tile, (h, w, 1))

    brick = resize(brick_img, (TILE, TILE))

    for x in range(h):
        for y in range(w):
            if env.grid[x][y] == 1:
                bg[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = brick

    # Rack slots
    for slot in env.rack_slots:
        x, y = slot
        if slot in env.filled_slots:
            bg[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = [0.2, 0.6, 0.2]
        else:
            bg[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = [0.8, 0.5, 0.2]

    # Start marker
    x, y = env.start
    bg[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = [0.1, 0.7, 0.7]

    # Finish marker
    x, y = env.goal
    bg[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = [0.9, 0.2, 0.7]

    return bg


# ── LAYER 2: OBJECTS ──
def build_objects(env, robot_pos):
    """
    Build the dynamic object layer (RGBA):
    - Uncollected boxes: box sprite
    - Robot: robot sprite (if position given)
    """
    h, w = env.size, env.size
    layer = np.zeros((h * TILE, w * TILE, 4))

    robot = resize(robot_img, (TILE, TILE))
    box = resize(box_img, (TILE, TILE))

    # Boxes (only those not yet collected)
    for b in env.boxes:
        if b in env.collected_boxes:
            continue
        x, y = b
        sx, sy = x * TILE, y * TILE
        ex, ey = (x + 1) * TILE, (y + 1) * TILE
        if box.shape[2] == 4:
            # Use sprite alpha to mask
            alpha_mask = box[:, :, 3:] > 0
            layer[sx:ex, sy:ey, :3] = np.where(
                alpha_mask, box[:, :, :3], layer[sx:ex, sy:ey, :3]
            )
            layer[sx:ex, sy:ey, 3] = np.where(
                alpha_mask.squeeze(), 1.0, layer[sx:ex, sy:ey, 3]
            )
        else:
            layer[sx:ex, sy:ey, :3] = box[:, :, :3]
            layer[sx:ex, sy:ey, 3] = 1.0

    # Robot
    if robot_pos is not None:
        x, y = robot_pos
        sx, sy = x * TILE, y * TILE
        ex, ey = (x + 1) * TILE, (y + 1) * TILE
        if robot.shape[2] == 4:
            alpha_mask = robot[:, :, 3:] > 0
            layer[sx:ex, sy:ey, :3] = np.where(
                alpha_mask, robot[:, :, :3], layer[sx:ex, sy:ey, :3]
            )
            layer[sx:ex, sy:ey, 3] = np.where(
                alpha_mask.squeeze(), 1.0, layer[sx:ex, sy:ey, 3]
            )
        else:
            layer[sx:ex, sy:ey, :3] = robot[:, :, :3]
            layer[sx:ex, sy:ey, 3] = 1.0

    return layer


# ── LAYER 3: OVERLAY ──
def _draw_line(layer, x0, y0, x1, y1, color, thickness=3):
    """
    Draw a line on the overlay layer using Bresenham's algorithm.
    (x0,y0) and (x1,y1) are pixel coordinates.
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    h, w = layer.shape[0], layer.shape[1]
    while True:
        for t in range(-thickness // 2, thickness // 2 + 1):
            for tt in range(-thickness // 2, thickness // 2 + 1):
                px, py = x0 + t, y0 + tt
                if 0 <= px < h and 0 <= py < w:
                    layer[px, py, :3] = color
                    layer[px, py, 3] = 1.0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def build_overlay(env, path, events):
    """
    Build the overlay layer (RGBA):
    - Path taken: semi-transparent purple line
    - Pickup events: gold
    - Delivery events: green
    """
    h, w = env.size, env.size
    layer = np.zeros((h * TILE, w * TILE, 4))

    # Path (semi-transparent purple line through tile centers)
    if path and len(path) > 1:
        half = TILE // 2
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            px0 = x0 * TILE + half
            py0 = y0 * TILE + half
            px1 = x1 * TILE + half
            py1 = y1 * TILE + half
            _draw_line(layer, px0, py0, px1, py1, [0.7, 0.4, 0.9], thickness=3)
    elif path and len(path) == 1:
        # Single point: just mark the tile center
        x, y = path[0]
        cx, cy = x * TILE + TILE // 2, y * TILE + TILE // 2
        for t in range(-2, 3):
            for tt in range(-2, 3):
                layer[cx + t, cy + tt, :3] = [0.7, 0.4, 0.9]
                layer[cx + t, cy + tt, 3] = 1.0

    # Events
    if events:
        for pos, kind in events.items():
            x, y = pos
            if kind == "pickup":
                layer[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE, :3] = [
                    1.0,
                    0.8,
                    0.0,
                ]
            else:  # delivery
                layer[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE, :3] = [
                    0.0,
                    1.0,
                    0.0,
                ]

    return layer


# ── COMPOSITOR ──
def compose(bg, obj, overlay):
    """
    Blend the three layers into a single final image.
    Uses alpha compositing with configurable opacity per layer.
    """
    final = bg.copy()

    def blend(base, top, alpha=0.8):
        return (1 - alpha) * base + alpha * top

    # Object layer (opacity 0.9)
    mask = np.any(obj > 0, axis=2)
    final[mask] = blend(final[mask], obj[mask, :3], 0.9)

    # Overlay layer (opacity 0.5)
    mask2 = np.any(overlay > 0, axis=2)
    final[mask2] = blend(final[mask2], overlay[mask2, :3], 0.5)

    return np.clip(final, 0.0, 1.0)


# ── RENDER STEP (Streamlit-safe) ──
def render_step(env, path_so_far, events, robot_pos, title=None):
    """
    Build all layers, composite them, and render to Streamlit.
    This is the main rendering function called per animation frame.
    """
    bg = build_background(env)
    obj = build_objects(env, robot_pos)
    overlay = build_overlay(env, path_so_far, events)

    final = compose(bg, obj, overlay)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(final, interpolation="nearest")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── FAST RENDER (no matplotlib, returns numpy array directly) ──
def render_to_array(env, robot_pos=None, path_so_far=None, events_so_far=None):
    """
    Build all layers, composite them, and return the final image as a numpy array.
    Much faster than render_step() because it avoids matplotlib figure creation.
    """
    bg = build_background(env)
    obj = build_objects(env, robot_pos)
    overlay = build_overlay(env, path_so_far or [], events_so_far or {})
    return compose(bg, obj, overlay)


# ── TOP-LEVEL DRAW FRAME (backward-compatible entry point) ──
def draw_frame(env, robot_pos=None, path_so_far=None, events_so_far=None,
               title=None):
    """
    Top-level rendering function.
    Uses fast st.image() with numpy array instead of slow matplotlib st.pyplot().
    """
    final = render_to_array(
        env=env,
        robot_pos=robot_pos,
        path_so_far=path_so_far or [],
        events_so_far=events_so_far or {},
    )
    # Convert float [0,1] to uint8 for st.image()
    img = (np.clip(final, 0, 1) * 255).astype(np.uint8)
    st.image(img, caption=title or "", use_container_width=True)
