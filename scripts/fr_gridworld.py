import subprocess

MAPS = ["6x6guided", "6x6sparse", "6x6liar", "6x6spelunky", "6x6spenalty10"]

if __name__ == "__main__":
    for map_ in MAPS:
        path = "../examples/fr_gridworld.py"
        logdir = "Results/{}-ep01".format(map_)
        subprocess.run(
            [
                "pipenv",
                "run",
                "python",
                path,
                "--logdir",
                logdir,
                "--map",
                map_,
                "--num-policy-checks=20",
                "--max-steps=4000",
                "--epsilon=0.4",
                "--epsilon-min=0.1",
                "--plot-save",
                # "--agent=count-based-q",
                # "--beta=1",
            ]
        )
