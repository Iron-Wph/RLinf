import numpy as np
import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("gymnasium")

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv  # noqa: E402


def _make_env_wrapper(*, single_arm: bool, active_arm: str = "left") -> RoboTwinEnv:
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg = OmegaConf.create(
        {
            "task_config": {
                "embodiment": ["franka-panda"],
                "single_arm": single_arm,
                "active_arm": active_arm,
            }
        }
    )
    env.center_crop = False
    env.task_name = "stack_blocks_two"
    env.is_franka_single_arm = single_arm
    env.active_arm = active_arm
    env.action_dim = 8 if single_arm else 16
    env.env_action_dim = env.action_dim
    env.vector_env_file = "/tmp/robotwin/envs/vector_env.py"
    env._last_raw_states = None
    env.venv = type(
        "DummyVectorEnv",
        (),
        {"args": {"single_arm": single_arm, "active_arm": active_arm}},
    )()
    return env


def _raw_observation(state=None):
    if state is None:
        state = np.arange(8, dtype=np.float32)
    return {
        "full_image": np.zeros((4, 6, 3), dtype=np.uint8),
        "left_wrist_image": np.full((4, 6, 3), 11, dtype=np.uint8),
        "right_wrist_image": np.full((4, 6, 3), 22, dtype=np.uint8),
        "state": state,
        "instruction": "stack the blocks",
    }


def test_franka_single_arm_observation_projects_legacy_dual_franka_state():
    env = _make_env_wrapper(single_arm=True)
    observation = _raw_observation(state=np.arange(16, dtype=np.float32))

    extracted = env._extract_obs_image([observation])

    assert extracted["states"].shape == (1, 8)
    assert np.all(extracted["states"][0].numpy() == np.arange(8, dtype=np.float32))


def test_franka_single_arm_action_can_expand_to_legacy_dual_franka_action():
    env = _make_env_wrapper(single_arm=True)
    env.venv.args["single_arm"] = False
    env.env_action_dim = 16
    env._last_raw_states = np.arange(16, dtype=np.float32)[None, :]

    env_action = env._adapt_actions_for_venv(
        np.ones((1, 2, 8), dtype=np.float32)
    )

    assert env_action.shape == (1, 2, 16)
    assert np.all(env_action[..., :8] == 1.0)
    assert np.all(env_action[:, :, 8:16] == np.arange(8, 16, dtype=np.float32))


@pytest.mark.parametrize(
    ("active_arm", "expected_pixel"),
    [("left", 11), ("right", 22)],
)
def test_single_arm_observation_keeps_only_active_wrist_camera(
    active_arm: str, expected_pixel: int
):
    env = _make_env_wrapper(single_arm=True, active_arm=active_arm)

    observation = env._extract_obs_image([_raw_observation()])

    assert observation["states"].shape == (1, 8)
    assert observation["wrist_images"].shape == (1, 1, 4, 6, 3)
    assert np.all(observation["wrist_images"][0, 0].numpy() == expected_pixel)


def test_single_arm_observation_defaults_to_left_arm():
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg = OmegaConf.create(
        {"task_config": {"embodiment": ["franka-panda"], "single_arm": True}}
    )
    env.center_crop = False
    env.task_name = "stack_blocks_two"
    env.is_franka_single_arm = True
    env.active_arm = "left"
    env.action_dim = 8
    env.env_action_dim = 8
    env.vector_env_file = "/tmp/robotwin/envs/vector_env.py"
    env._last_raw_states = None

    observation = env._extract_obs_image([_raw_observation()])

    assert observation["wrist_images"].shape == (1, 1, 4, 6, 3)
    assert np.all(observation["wrist_images"][0, 0].numpy() == 11)


def test_single_arm_observation_rejects_legacy_dual_arm_state_vector():
    env = _make_env_wrapper(single_arm=True)
    observation = _raw_observation(state=np.arange(14, dtype=np.float32))

    with pytest.raises(ValueError, match="expected shape"):
        env._extract_obs_image([observation])


def test_dual_arm_observation_keeps_both_wrist_cameras():
    env = _make_env_wrapper(single_arm=False)

    observation = env._extract_obs_image(
        [_raw_observation(state=np.arange(16, dtype=np.float32))]
    )

    assert observation["states"].shape == (1, 16)
    assert observation["wrist_images"].shape == (1, 2, 4, 6, 3)
    assert np.all(observation["wrist_images"][0, 0].numpy() == 11)
    assert np.all(observation["wrist_images"][0, 1].numpy() == 22)


def test_single_arm_action_validation_uses_policy_dimension():
    env = _make_env_wrapper(single_arm=True)

    env._validate_action_dim(np.zeros((2, 1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 8, got 14"):
        env._validate_action_dim(np.zeros((2, 1, 14), dtype=np.float32))
