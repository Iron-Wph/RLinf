import numpy as np
import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("gymnasium")

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv  # noqa: E402


def _make_env_wrapper(*, single_arm: bool, active_arm: str = "right") -> RoboTwinEnv:
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg = OmegaConf.create(
        {"task_config": {"single_arm": single_arm, "active_arm": active_arm}}
    )
    env.center_crop = False
    env.task_name = "stack_blocks_two"
    env.action_dim = 8
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


def test_single_arm_observation_rejects_legacy_dual_arm_state_vector():
    env = _make_env_wrapper(single_arm=True)
    observation = _raw_observation(state=np.arange(16, dtype=np.float32))

    with pytest.raises(ValueError, match="does not support task_config.single_arm"):
        env._extract_obs_image([observation])


def test_dual_arm_observation_keeps_both_wrist_cameras():
    env = _make_env_wrapper(single_arm=False)

    observation = env._extract_obs_image([_raw_observation()])

    assert observation["wrist_images"].shape == (1, 2, 4, 6, 3)
    assert np.all(observation["wrist_images"][0, 0].numpy() == 11)
    assert np.all(observation["wrist_images"][0, 1].numpy() == 22)


def test_single_arm_action_validation_uses_vector_env_dimension():
    env = _make_env_wrapper(single_arm=True)

    env._validate_action_dim(np.zeros((2, 1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 8, got 14"):
        env._validate_action_dim(np.zeros((2, 1, 14), dtype=np.float32))
