from ray.tune import Analysis
import pandas as pd

# Specify the results directory
results_dir = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/hyperparameter_Tuning"

# Load the analysis object
analysis = Analysis(results_dir)

# Retrieve all trial dataframes
df = analysis.dataframe()

# Filter successful trials
successful_trials = df[df['trial_status'] == 'TERMINATED']

# Display the successful trials
print("Successful trials:")
print(successful_trials)

## Optionally, save the successful trials to a CSV file
#successful_trials.to_csv("successful_trials.csv", index=False)
#
## Access the best trial based on a specific metric
#best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
#print("Best trial:")
#print(best_trial)
#
## Access the best config
#best_config = analysis.get_best_config(metric="val_loss", mode="min", scope="all")
#print("Best config:")
#print(best_config)
#
## Access checkpoints from successful trials
#for trial in successful_trials['logdir']:
#    checkpoint_path = analysis.get_best_checkpoint(trial, metric="val_loss", mode="min")
#    print(f"Best checkpoint for trial {trial}: {checkpoint_path}")