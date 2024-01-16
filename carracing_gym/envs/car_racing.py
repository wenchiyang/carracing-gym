"""
Inherits the
[CarRacing environment](https://www.gymlibrary.dev/environments/box2d/car_racing/)
(gym.envs.box2d.car_racing) but with discrete control instead of continuous one
and a simpler observation space.

State consists of four consecutive images of STATE_W x STATE_H pixels. If the
render mode is "state_pixels", the states are in colors. If the render mode is
"gray", the states are in grayscale.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The track in
all episodes are the same, generated at initialization.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.
"""
import math
import numpy as np

from gym import spaces
from gym.spaces.discrete import Discrete

import pyglet
import random
from random import sample
from gym.utils import seeding

pyglet.options["debug_gl"] = False
from pyglet import gl

# the original env
from gym.envs.box2d.car_racing import CarRacing as GymCarRacing
from gym.envs.box2d.car_dynamics import Car

import platform


STATE_W = 48  # less than Atari 160x192
STATE_H = 48
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

os_type = platform.system()
if os_type == "Darwin":
    pass
elif os_type == "Linux":
    from pyvirtualdisplay import Display
    Display(visible=0, size=(WINDOW_W, WINDOW_H)).start()

ACTION_LOOKUP = {
    0: "do nothing",
    1: "accelerate",
    2: "brake",
    3: "turn left",
    4: "turn right",
}

ACTION_CONT = [
    [0.0, 0.0, 0.0],  # Do-Nothing
    [0.0, 1.0, 0.0],  # Accelerate
    [0.0, 0.0, 0.8],  # Brake
    [-1.0, 0.0, 0.0],  # Turn_left
    [1.0, 0.0, 0.0],  # Turn_right
]


SCALE = 6.0  # Track scale
TRACK_RAD = 500 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 1200 / SCALE  # Game over boundary #2000
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom #2.7
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
REWARD_COLOR = [0.0, 0.0, 0.0]

MY_BAR_COLOR = -1.0
MY_GRASS_COLOR = 0.26464
MY_ROAD_COLOR = -0.20313
MY_CAR_COLOR = -0.52347

NUM_STACKED_IMG = 4


class CarRacing(GymCarRacing):
    metadata = {
        "render.modes": ["human", "state_pixels", "gray"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=1, seed=None, render_mode=True):
        super(CarRacing, self).__init__()
        self.verbose = verbose
        if seed is not None:
            self.seed(seed)
        # Fix the render mode for the step(), reset(), etc
        self.render_mode = render_mode


        # observation space
        self.stack = []  # the state
        self.num_stacked_img = NUM_STACKED_IMG
        # dim of the image
        self.height = 48

        if self.render_mode == "gray":
            self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(self.num_stacked_img, self.height, self.height),
                dtype=np.float32,
            )
        elif self.render_mode == "state_pixels":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.num_stacked_img, STATE_H, STATE_W),
                dtype=np.float32,
            )
        elif self.render_mode == "human":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )

        # action space
        self.action_size = 5
        self.action_space = Discrete(self.action_size)

        # configure the shape of the track
        self.spikyness = [0.3333, 1]
        self.n_rewardpoints = 500
        self.n_corners = 12
        # create the map once and reuse the same map later
        self._create_map()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _create_map(self):
        self.road_poly_storage = []
        self.track_storage = []
        self.road_poly_reward_storage = []

        self._destroy()
        self.road_poly = []
        self.road_poly_reward = []
        while True:
            success, maps_info = self._create_track()
            if success:
                self.road_poly_storage.append(maps_info[0])
                self.track_storage.append(maps_info[1])
                self.road_poly_reward_storage.append(maps_info[2])
                self.n_total_tiles = min(len(self.road_poly), self.n_rewardpoints)
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

    def _create_track(self):
        CHECKPOINTS = self.n_corners

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            # rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            rad = self.np_random.uniform(
                TRACK_RAD * self.spikyness[0], TRACK_RAD * self.spikyness[1]
            )

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            # track_temp.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False, None  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False, None

        # Red-white border on hard turns
        border = [False] * len(track)
        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            # c = 0.01 * (i % 3)
            c = 0
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
        self.track = track
        road_poly_reward = sample(
            self.road_poly, min(len(self.road_poly), self.n_rewardpoints)
        )
        self.road_poly_reward = [vs for vs, _ in road_poly_reward]
        return True, (self.road_poly, self.track, self.road_poly_reward)

    def _reset_track(self):
        """
        Reset the track. Intended to be used by reset()
        :return:
        """
        ind = 0

        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.tile_visited_count_previous = 0
        self.t = 0.0
        self.road_poly = []

        self.road_poly = self.road_poly_storage[ind]
        self.track = self.track_storage[ind]
        self.road_poly_reward = self.road_poly_reward_storage[ind]

        # Create tiles
        for (vertices, color) in self.road_poly:
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            t.color = color
            # t.road_visited = False
            t.road_visited = True
            if vertices in self.road_poly_reward:
                t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road.append(t)

    def reset(self):
        self._reset_track()
        self.car = Car(self.world, *self.track[0][1:4])
        return self.step(None)[0]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        if self.car:
            self.car.destroy()

    def step(self, action):
        if action is not None:
            # convert the discrete action to a continuous one
            action = ACTION_CONT[action]
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        state_gray = self.render("gray")
        self.state = state_gray

        if action is None:
            self.stack = [state_gray] * self.num_stacked_img
        else:
            self.stack.pop(0)
            self.stack.append(state_gray)

        assert len(self.stack) == self.num_stacked_img

        step_reward = 0
        done = False
        info = {
            "is_success": False,
        }
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            visit_new_reward = bool(
                self.tile_visited_count - self.tile_visited_count_previous
            )
            self.tile_visited_count_previous = self.tile_visited_count
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            # give step_reward only when the car pass by cirtain points
            car_x, car_y = self.car.hull.position.x, self.car.hull.position.y
            ####
            # success!
            if self.tile_visited_count == self.n_total_tiles:
                done = True
                info["is_success"] = True
            x, y = self.car.hull.position
            #
            info["out_of_field"] = False
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100
                info["is_success"] = False
                info["out_of_field"] = True

        return np.array(self.stack), step_reward, done, info

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "gray"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        # zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        zoom = ZOOM * SCALE
        # zoom = 1.0
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        # angle = 0
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, draw_particles=False)

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "gray" or mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )

        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        if mode == "state_pixels":
            return arr
        elif mode == "gray":
            arr = self.rgb2gray(arr)
            return arr

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128.0 - 1.0

            gray[np.isclose(gray, MY_BAR_COLOR, rtol=1e-03, atol=1e-03)] = 1
            gray[np.isclose(gray, MY_GRASS_COLOR, rtol=1e-03, atol=1e-03)] = 0.6
            gray[np.isclose(gray, MY_ROAD_COLOR, rtol=1e-03, atol=1e-03)] = -0.1
            gray[np.isclose(gray, MY_CAR_COLOR, rtol=1e-03, atol=1e-03)] = -1

            return gray

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_indicators(self, W, H):
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

    def get_action_lookup(self):
        return ACTION_LOOKUP


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array(0)

    def key_press(k, mod):
        global restart
        global a
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a = np.array(3)
        if k == key.RIGHT:
            a = np.array(4)
        if k == key.UP:
            a = np.array(1)
        if k == key.DOWN:
            a = np.array(2)

    def key_release(k, mod):
        global a
        if k == key.LEFT or  k == key.RIGHT or k == key.UP or k == key.DOWN:
            a = np.array(0)

    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str([f"{a}"]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
