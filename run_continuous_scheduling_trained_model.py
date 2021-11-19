import os

from mahi_api.interfaces.v2.red_time import RedTime
from mahi_api.interfaces.v2.shift import Shift
from mahi_api.interfaces.v2.task import Task
from scheduling_env.envs.continuous_scheduling_environmentV1 import (
    ContinuousSchedulingEnvV1,
)

from utils.utils import ALGOS, get_latest_run_id
from stable_baselines3.common.utils import set_random_seed


def main():  # noqa: C901
    seed = 0
    env_id = "scheduling-cont-v1"
    algo = "ppo"
    folder = "logs/"

    exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    print(f"Loading latest experiment, id={exp_id}")
    log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")
    set_random_seed(seed)

    shift = Shift(
        id="1",
        start_time=1637181427,
        end_time=1637192227,
        latitude=1.0,
        longitude=1.0,
        step_delta=1080,
        n_step=10,
        weather=[],
        schedule=[],
        tasks={
            "task_0": Task(
                id="task_0",
                total_time=30 * 60,
                remaining_time=30 * 60,
                progress=0.0,
            ),
            "task_1": Task(
                id="task_1",
                total_time=30 * 60,
                remaining_time=30 * 60,
                progress=0.0,
            ),
        },
        redtimes={
            "red_0": RedTime(
                id="red_0", start_time=1637182200, end_time=1637184000, prob=0.10
            ),
            "red_1": RedTime(
                id="red_1", start_time=1637188200, end_time=1637189400, prob=0.30
            ),
        },
    )

    env = ContinuousSchedulingEnvV1()
    model = ALGOS[algo].load(
        model_path,
        env=env,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
        seed=seed,
    )

    obs = env.reset(shift)
    predicted_schedule = None

    try:
        state = None

        env.render("human")
        for _ in range(100):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, infos = env.step(action)

            if done:
                break

            predicted_schedule = infos["schedule"]
            env.render("human")

    except KeyboardInterrupt:
        pass

    print(predicted_schedule.get_schedule_array(shift.n_step))

    env.close()


if __name__ == "__main__":
    main()
