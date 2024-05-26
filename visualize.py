from ray.tune import ExperimentAnalysis
import pandas as pd
import os
import shutil
import re
from collections import defaultdict

def read_save_sort_raytune_data(results_dir = "C:\\Users\\roibl\\Downloads\\hyperparameter_Tuning\\train_cifar_2024-05-19_13-20-47"):

    # Load the analysis object
    analysis = ExperimentAnalysis(results_dir)

    # Retrieve all trial dataframes
    df = analysis.dataframe()

    config_columns = [col for col in df.columns if col.startswith('config/')]

    # Erstelle ein neues DataFrame mit den gefilterten Spalten
    config_df = df[config_columns]

    # Umbenennen der Spalten für einfacheren Zugriff
    config_df.columns = [col.split('/')[-1] for col in config_df.columns]

    # Finde die maximale seq_len für jede Methode
    config_df['max_seq_len'] = config_df.groupby('method')['seq_len'].transform('max')

    # Filtere das DataFrame, um nur die Zeilen mit der maximalen seq_len zu behalten
    max_seq_len_df = config_df[config_df['seq_len'] == config_df['max_seq_len']]

    # Entferne die Hilfsspalte 'max_seq_len'
    max_seq_len_df = max_seq_len_df.drop(columns=['max_seq_len'])

    # Sortiere das DataFrame nach der Spalte 'method'
    max_seq_len_df = max_seq_len_df.sort_values(by='method')

    # Ergebnis anzeigen
    print(max_seq_len_df)

    # Optionally, save the successful trials to a CSV file
    max_seq_len_df.to_csv("successful_trials.csv", index=False)
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

def remove_duplicate_runs(directory):
    count_deleted_folders = 0
    # Regex, um den relevanten Teil des Ordnernamens zu extrahieren
    pattern = re.compile(r'^[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')

    # Dictionary, um die Ordner nach ihrem relevanten Namen zu gruppieren
    folders = defaultdict(list)

    # Alle Ordner im Verzeichnis durchlaufen
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match:
            key = match.group(1)
            folders[key].append(folder_name)

    # Ordner löschen, die sich nur im Timestamp und Prefix unterscheiden
    for key, folder_list in folders.items():
        # Behalte einen Ordner und lösche den Rest
        for folder_name in folder_list[1:]:
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                count_deleted_folders += 1
                #print(f"Lösche Ordner: {folder_path}")
                shutil.rmtree(folder_path)  # Verwenden Sie shutil.rmtree(folder_path) für nicht-leere Ordner
    
    print(f"Number of runs: {len(folders.items())}, deleted_folders: {count_deleted_folders}")

if __name__ == "__main__":
    remove_duplicate_runs("C:\\Users\\roibl\\Downloads\\hyperparameter_Tuning\\train_cifar_2024-05-19_13-20-47")