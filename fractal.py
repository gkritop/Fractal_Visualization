import numpy as np
import pygame
from numba import njit, prange

# Constants
WIDTH, HEIGHT = 1200, 800
MAX_ITER = 200
ESCAPE_RADIUS = 100.0  # Squared value (10*10) for smoother gradients

# We define RGB stops to interpolate between.
# Stop 0: Deep Purple/Black (Background)
C1 = np.array([10, 0, 30], dtype=np.float64)
# Stop 1: Bright Blue/Cyan
C2 = np.array([0, 180, 220], dtype=np.float64)
# Stop 2: Pale Yellow/White (High Luminance)
C3 = np.array([255, 255, 200], dtype=np.float64)


@njit(inline="always")
def interpolate_color(t):
    # Wrap t so it cycles smoothly
    t = t % 1.0

    if t < 0.5:
        # Interpolate between C1 and C2
        local_t = t * 2.0
        r = C1[0] + (C2[0] - C1[0]) * local_t
        g = C1[1] + (C2[1] - C1[1]) * local_t
        b = C1[2] + (C2[2] - C1[2]) * local_t
    else:
        # Interpolate between C2 and C3
        local_t = (t - 0.5) * 2.0
        r = C2[0] + (C3[0] - C2[0]) * local_t
        g = C2[1] + (C3[1] - C2[1]) * local_t
        b = C2[2] + (C3[2] - C2[2]) * local_t

    return int(r), int(g), int(b)


@njit(parallel=True, fastmath=True)
def compute_fractal(
    width: int,
    height: int,
    zoom: float,
    offset_x: float,
    offset_y: float,
    julia_re: float,
    julia_im: float,
    is_mandelbrot: bool,
    time_shift: float,
):
    buffer = np.empty((height, width, 3), dtype=np.uint8)

    scale_x = 3.5 / (zoom * width)
    scale_y = 2.0 / (zoom * height)
    center_x = width / 2.0
    center_y = height / 2.0

    for y in prange(height):
        for x in range(width):
            # Map pixel coordinate to complex plane
            real = (x - center_x) * scale_x + offset_x
            imag = (y - center_y) * scale_y + offset_y

            # Starting values based on Fractal Type
            if is_mandelbrot:
                zx, zy = 0.0, 0.0
                cx, cy = real, imag
            else:
                zx, zy = real, imag
                cx, cy = julia_re, julia_im

            iteration = 0

            # Check radius squared
            while zx * zx + zy * zy <= ESCAPE_RADIUS and iteration < MAX_ITER:
                xtemp = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = xtemp
                iteration += 1

            # Coloring Logic
            if iteration == MAX_ITER:
                # Inside the set: Draw Black
                buffer[y, x, 0] = 0
                buffer[y, x, 1] = 0
                buffer[y, x, 2] = 0
            else:
                # Smooth coloring
                log_zn = np.log(zx * zx + zy * zy) / 2
                nu = np.log(log_zn / np.log(2)) / np.log(2)
                iteration_smooth = iteration + 1 - nu

                # 0.03 controls the "frequency" of the bands
                t = iteration_smooth * 0.03 + time_shift

                r, g, b = interpolate_color(t)

                buffer[y, x, 0] = r
                buffer[y, x, 1] = g
                buffer[y, x, 2] = b

    return buffer


def main():
    # Initialization
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fractal Explorer")
    clock = pygame.time.Clock()

    # State Variables
    zoom = 1.0
    offset_x = -0.75
    offset_y = 0.0
    time_shift = 0.0

    is_mandelbrot = True
    auto_zoom = False

    # Track mouse for Julia set morphology
    pygame.mouse.get_rel()

    print("--- Controls ---")
    print("[Mouse]: Pan around (Click & Drag)")
    print("[Scroll]: Zoom In/Out")
    print("[Space]: Switch between Mandelbrot and Julia Set")
    print("[A]: Toggle Auto-Zoom")
    print("[R]: Reset View")

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom centered on view
                zoom_factor = 1.0 + event.y * 0.1
                zoom *= zoom_factor

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_mandelbrot = not is_mandelbrot

                    if is_mandelbrot:
                        offset_x, offset_y = -0.75, 0.0

                elif event.key == pygame.K_r:
                    zoom, offset_x, offset_y = 1.0, -0.75, 0.0
                    auto_zoom = False

                elif event.key == pygame.K_a:
                    auto_zoom = not auto_zoom

        # 2. Update State
        time_shift += 0.002  # Color cycling speed

        if auto_zoom:
            zoom *= 1.01

        # Interactive Julia Set: Map mouse position to Complex Plane
        mx, my = pygame.mouse.get_pos()
        # Normalized coordinates (-1 to 1)
        j_re = (mx / WIDTH - 0.5) * 2.0
        j_im = (my / HEIGHT - 0.5) * 2.0

        # Panning Logic
        if pygame.mouse.get_pressed()[0]:
            rel_x, rel_y = pygame.mouse.get_rel()

            offset_x -= rel_x / (300.0 * zoom)
            offset_y -= rel_y / (300.0 * zoom)
        else:
            pygame.mouse.get_rel()

        # 3. Compute Frame
        pixel_data = compute_fractal(
            WIDTH,
            HEIGHT,
            zoom,
            offset_x,
            offset_y,
            j_re,
            j_im,
            is_mandelbrot,
            time_shift,
        )

        # 4. Render
        # Transpose needed because Pygame expects (Width, Height, 3)
        surface = pygame.surfarray.make_surface(np.transpose(pixel_data, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        # HUD Update
        mode_text = "Mandelbrot" if is_mandelbrot else f"Julia ({j_re:.2f}, {j_im:.2f})"
        pygame.display.set_caption(f"Fractal Explorer | {mode_text} | Zoom: {zoom:.1e}")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
