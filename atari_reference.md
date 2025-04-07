# Atari Environment Reference

## Action Space
| Value | Meaning      | Value | Meaning      | Value | Meaning       |
|-------|--------------|-------|--------------|-------|---------------|
| 0     | NOOP         | 6     | UPRIGHT      | 12    | LEFTFIRE      |
| 1     | FIRE         | 7     | UPLEFT       | 13    | DOWNFIRE      |
| 2     | UP           | 8     | DOWNRIGHT    | 14    | UPRIGHTFIRE   |
| 3     | RIGHT        | 9     | DOWNLEFT     | 15    | UPLEFTFIRE    |
| 4     | LEFT         | 10    | UPFIRE       | 16    | DOWNRIGHTFIRE |
| 5     | DOWN         | 11    | RIGHTFIRE    | 17    | DOWNLEFTFIRE  |

- Default: Most environments use a smaller subset of legal actions
- Full action space can be enabled with `full_action_space=True`

## Continuous Action Space
- 3-dimensional vector: (radius, theta, fire)
  - radius: [0.0, 1.0]
  - theta: [-np.pi, np.pi]
  - fire: [0.0, 1.0]
- Activated by setting `continuous=True`
- Actions thresholded using `continuous_action_threshold`

## Observation Space
| Type        | Parameter             | Shape           | Data Type |
|-------------|------------------------|-----------------|-----------|
| RGB         | `obs_type="rgb"`       | (210, 160, 3)   | np.uint8  |
| Grayscale   | `obs_type="grayscale"` | (210, 160)      | np.uint8  |
| RAM         | `obs_type="ram"`       | (128,)          | np.uint8  |

## Stochasticity Methods
1. **Sticky Actions**:
   - Probability that previous action repeats instead of new action
   - v0, v5: 25% probability (`repeat_action_probability=0.25`)
   - v4: 0% probability (`repeat_action_probability=0.0`)

2. **Frameskipping**:
   - Deterministic: Action repeats for fixed number of frames
   - Stochastic: Action repeats for random frames in range
   - Configure with `frameskip=N` or `frameskip=(min, max)`

## Common Arguments
| Argument                      | Type            | Description                                     |
|-------------------------------|----------------|-------------------------------------------------|
| `mode`                        | int            | Game mode, varies by environment                |
| `difficulty`                  | int            | Game difficulty, varies by environment          |
| `obs_type`                    | str            | Observation type: "ram", "rgb", "grayscale"     |
| `frameskip`                   | int or tuple   | Number of frames to repeat action               |
| `repeat_action_probability`   | float          | Probability of repeating previous action        |
| `full_action_space`           | bool           | Use all legal console actions                   |
| `continuous`                  | bool           | Use continuous action space                     |
| `render_mode`                 | str            | "human" or "rgb_array"                          |

## Version Comparison
| Version | frameskip  | repeat_action_probability | Namespace |
|---------|------------|---------------------------|-----------|
| v0      | (2, 5)     | 0.25                      | -         |
| v4      | (2, 5)     | 0.0                       | -         |
| v5      | 4          | 0.25                      | ALE/      |

## Environment Variants for IceHockey
| Environment ID              | obs_type | frameskip | repeat_action_probability |
|-----------------------------|----------|-----------|---------------------------|
| ALE/IceHockey-v5            | "rgb"    | 4         | 0.25                      |
| ALE/IceHockey-ram-v5        | "ram"    | 4         | 0.25                      |
| IceHockey-v4                | "rgb"    | (2, 5)    | 0.0                       |
| IceHockeyDeterministic-v4   | "rgb"    | 4         | 0.0                       |
| IceHockeyNoFrameskip-v4     | "rgb"    | 1         | 0.0                       |

## Best Practices
- Use v5 environments (ALE namespace) as they follow best practices from research
- Customize environment behavior using arguments to `gymnasium.make()`
- For DQN training, use frameskipping, sticky actions, and frame stacking
