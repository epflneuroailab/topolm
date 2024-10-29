# conda activate /work/upschrimpf1/mehrer/code/20240709_faciotopy_GLM/faciotopy_GLM_env
################################
# fMRI analysis of Hauptman et al. 2024
################################

################################
# 1. Download data
################################
# Please download the pre-processed fMRI data from Hauptman et al. 2024
# (https://doi.org/10.1101/2023.08.23.552701) from the openicpsr platform 
# (https://doi.org/10.3886/E198163V2) and adjust the path 
# ("Hauptman2024_data_dir") to the data.

# each subject folder contains structural data
# lh.32k_fs_LR.hcp_inflated.surf.gii and rh.32k_fs_LR.hcp_inflated.surf.gii  
# lh.32k_fs_LR.midthickness.surf.gii and rh.32k_fs_LR.midthickness.surf.gii

# each subject folder contains 8 functional runs
# each run of each sub contains the following functional (the first two are our main interest here):
# lh.32k_fs_LR.surfed_data.func.gii and rh.32k_fs_LR.surfed_data.func.gii
# lh.32k_fs_LR.mask.shape.gii and rh.32k_fs_LR.mask.shape.gii

# each sub contains a logfile.csv with the following columns for all the 8 runs combined:
# Run, Restart, Trial, Onset, Cat, Subsem, Word1, Word2, Ans.given, Py.RT, Cedrus.RT

# Please, NOTE that the log files provided in the dataset do not explicitely describe baseline trials. 
# We add them to the log files using the present script. Here is an EXAMPLE (sub-22, run-01) 
# of the required changes. 

# The first run of sub-22 is 456 seconds long. 
# The logfile shows 72 trials and since each trial is 4 seconds long,
# the experimental trials would take up 72 * 4 = 288 seconds.

# However, the experiment is organized into blocks, where each block is 16 seconds long
# (4 trials per block), followed by 10 seconds of rest.
# Thus, there are (16 + 10 =) 26 seconds per block.
# The logfile only contains the onsets of a trial and does not explicitly 
# specify the offset of the trials, nor the onsets or offsets of the rest periods.

# After adjusting the log-files (computing offsets and adding rest periods),
# the run still has 72 condition trials (each 4 seconds long) and now has 17 
# rest periods (each 10 seconds long) resulting in a total run length of
# 72 * 4 + 17 * 10 = 458 seconds. 

# The log files indicates the onset of the last trial to be at 456 seconds, which is 
# 2 seconds before the end of the run. Given that the first trial starts at 2 
# seconds, it appears that the volume acquisition time is always at the center of the trial.

################################
# 2. Transform log files
################################
# With the following script we add baseline trials to the log files of the sighted subjects. Based 
# on the updated log files, we compute create a GLM to estimate the BOLD response to the experimental
# conditions: verbs / nouns. 

import pandas as pd
import numpy as np
from os.path import join as opj

Hauptman2024_data_dir = '/work/upschrimpf1/mehrer/datasets/Hauptman_2023/DataZip'
sighted_subs = [f'PTP_{i:02d}' for i in range(22, 44)] # sightes subjects: 22-43

for sub in sighted_subs:

    # load csv
    df = pd.read_csv(opj(Hauptman2024_data_dir, f'{sub}/logfile.csv'))

    # compute offset
    df['Offset'] = df['Onset'] + 4.0  # Each trial lasts 4 seconds

    new_rows = []
    for i in range(len(df)):
        new_rows.append({
            'Run': df.loc[i, 'Run'],
            'Restart': df.loc[i, 'Restart'],
            'Trial': df.loc[i, 'Trial'],
            'Onset': df.loc[i, 'Onset'],
            'Duration': 4,
            'Cat': df.loc[i, 'Cat'],
            'Subsem': df.loc[i, 'Subsem'],
            'Word1': df.loc[i, 'Word1'],
            'Word2': df.loc[i, 'Word2'],
            'Ans.given': df.loc[i, 'Ans.given'],
            'Py.RT': df.loc[i, 'Py.RT'],
            'Cedrus.RT': df.loc[i, 'Cedrus.RT']
        })
        
        # not last trial
        if i < len(df) - 1:
            # Calculate gap between current trial's offset and next trial's onset
            next_onset = df.loc[i + 1, 'Onset']
            current_offset = df.loc[i, 'Offset']
            gap = next_onset - df.loc[i, 'Onset']
            
            # If the gap is approx. 14 seconds (4 second trial + 10 second baseline), insert a baseline trial
            if np.isclose(gap, 14, atol=0.5):
                baseline_onset = current_offset
                baseline_duration = 10  # Baseline trial duration: 10 seconds
                new_rows.append({
                    'Run': df.loc[i, 'Run'],
                    'Restart': df.loc[i, 'Restart'],
                    'Trial': 'NA',  # adjusted below
                    'Onset': baseline_onset,
                    'Duration': baseline_duration,
                    'Cat': 'baseline',
                    'Subsem': np.nan,
                    'Word1': np.nan,
                    'Word2': np.nan,
                    'Ans.given': np.nan,
                    'Py.RT': np.nan,
                    'Cedrus.RT': np.nan
                })

    new_df = pd.DataFrame(new_rows)

    new_df['Trial'] = new_df.groupby(['Run']).cumcount() + 1

    new_df.to_csv(f'/work/upschrimpf1/mehrer/datasets/Hauptman_2023/DataZip/{sub}/logfile_with_baseline.csv', index=False)


