## A simpler CarRacing env



| property          | value                                               |
|-------------------|-----------------------------------------------------|
| Action Space      | `Discrete(5, int)`                                  |
| Observation Shape | `Box(4, 48, 48)` for                                |
| Observation Range | (-1, 1) for `gray` <br/>(0, 255) for `state_pixels` |
| Import            | `gym.make("CarRacingPLS-v1")`                       |


Inherits the
[CarRacing environment](https://www.gymlibrary.dev/environments/box2d/car_racing/) 
(gym.envs.box2d.car_racing) but with discrete control instead of continuous one
and a simpler observation space.

State consists of four consecutive images of STATE_W x STATE_H pixels. If the
render mode is `state_pixels`, the states are in colors. If the render mode is
`gray`, the states are in grayscale.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The track in
all episodes are the same, generated at initialization.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.



## Installation

```shell script
pip install -e .
```

## Configure the environment
Parameters for the environment.
  - `seed`: the seed of the environment. 
  - `verbose`: if the value is one, print debug information.
  - `render_mode`: determines the state space. If the value is `state_pixels` then the states are in colors.
If the value is `gray`, then the states are in grayscale.

