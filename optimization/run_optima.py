import argparse
import json
import optuna
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--results_csv', type=str, default=os.path.join('optimization', 'runs_in.csv'))
argparser.add_argument('--n_trials', type=int, default=8)
args = argparser.parse_args()

# Load existing results
existing_results = []
with open(args.results_csv, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        name, config, result = line.strip().split('|')
        config = json.loads(config)
        result = float(result)
        existing_results.append((name, config, result))

# Create a new study
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

distributions = {
    "lr": optuna.distributions.FloatDistribution(high=0.0001, log=True, low=1e-06, step=None),
    "warmup_steps": optuna.distributions.CategoricalDistribution([0, 2000, 4000, 6000, 8000]),
    "grokfast_lambda": optuna.distributions.FloatDistribution(0.0, 3.0),
    "grokfast_alpha": optuna.distributions.FloatDistribution(0.9, 1.0),
    "dropout": optuna.distributions.FloatDistribution(0.0, 0.3),
    "beta2": optuna.distributions.FloatDistribution(0.95, 0.9999),
}

# Add completed trials with their configs and results
for name, config, result in existing_results:
    # print(f"Adding completed trial with config: {config} and result: {result}")
    if result == 'fail':
        trial = optuna.trial.create_trial(
            params=config,
            distributions=distributions,
            state=optuna.trial.TrialState.FAIL
        )
    else:
        trial = optuna.trial.create_trial(
            params=config,
            distributions=distributions,
            value=result
        )
    study.add_trial(trial)

def get_repr(f):
    if isinstance(f, float):
        if f >= 0.01:
            return float(f"{f:.4f}")
        elif f == 0:
            return 0
    elif isinstance(f, float):
        # return truncated scientific notation
        return float(f"{f:.2e}")
    return f

for _ in range(args.n_trials):
    next_trial = study.ask()
    next_config = study.ask(distributions)
    truncated_params = {k: get_repr(v) for k, v in next_config.params.items()}
    print(str(truncated_params).replace("'", '"'))

    argstring = []
    for config_name, config_value in next_config.params.items():
        config_value = get_repr(config_value)
        argstring.append(f"--{config_name} {config_value}")

    print(' '.join(argstring))
