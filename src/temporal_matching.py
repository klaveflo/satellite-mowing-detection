"""
Temporal matching of satellite images to mowing events.
Lifted from Feature_Engineering.ipynb (PA2 project) and parameterized
so the same logic can be reused for Sentinel-1 matching.

Default windows (in days) match the PA2 configuration:
  BEFORE_FAR:  -20 to -9  days relative to event
  BEFORE_NEAR:  -8 to -3  days relative to event
  AFTER:        +1 to +7  days relative to event
"""

import os
import glob
import pandas as pd
from datetime import timedelta
from tqdm import tqdm


def load_sentinel_dates(sentinel_dir: str, extension: str = '.tif') -> pd.DataFrame:
    """
    Scan a directory of Sentinel images named YYYY-MM-DD<ext> and return
    a DataFrame with columns: file, date, filename.
    """
    files = sorted(glob.glob(os.path.join(sentinel_dir, f'*{extension}')))
    records = []
    for f in files:
        filename = os.path.basename(f).replace(extension, '')
        try:
            date = pd.to_datetime(filename)
            records.append({'file': f, 'date': date, 'filename': filename})
        except Exception as e:
            print(f"Warning: Cannot parse date from '{filename}': {e}")
    return pd.DataFrame(records)


def find_triplet_matches(
    masks_overview: pd.DataFrame,
    sentinel_df: pd.DataFrame,
    before_far_min: int = 9,
    before_far_max: int = 20,
    before_near_min: int = 3,
    before_near_max: int = 8,
    after_min: int = 1,
    after_max: int = 7,
) -> pd.DataFrame:
    """
    For each mowing event in masks_overview, find the closest Sentinel image
    in each of the three temporal windows.

    Parameters
    ----------
    masks_overview : DataFrame with columns 'date' (datetime) and 'date_str' and 'filename'
    sentinel_df    : DataFrame from load_sentinel_dates()
    *_min / *_max  : window boundaries in days (positive integers)

    Returns
    -------
    DataFrame with one row per matched event, columns:
        event_date, event_date_str, mask_file,
        before_far_file, before_near_file, after_file
    """
    matches = []

    for _, row in tqdm(masks_overview.iterrows(), total=len(masks_overview), desc="Temporal matching"):
        event_date = row['date']

        after_window = sentinel_df[
            (sentinel_df['date'] >= event_date + timedelta(days=after_min)) &
            (sentinel_df['date'] <= event_date + timedelta(days=after_max))
        ]
        before_near_window = sentinel_df[
            (sentinel_df['date'] >= event_date - timedelta(days=before_near_max)) &
            (sentinel_df['date'] <= event_date - timedelta(days=before_near_min))
        ]
        before_far_window = sentinel_df[
            (sentinel_df['date'] >= event_date - timedelta(days=before_far_max)) &
            (sentinel_df['date'] <= event_date - timedelta(days=before_far_min))
        ]

        if len(after_window) > 0 and len(before_near_window) > 0 and len(before_far_window) > 0:
            img_after      = after_window.iloc[0]
            img_before_near = before_near_window.iloc[-1]
            img_before_far  = before_far_window.iloc[-1]

            matches.append({
                'event_date':      event_date,
                'event_date_str':  row['date_str'],
                'mask_file':       row['filename'],
                'before_far_file': img_before_far['filename'] + '.tif',
                'before_near_file': img_before_near['filename'] + '.tif',
                'after_file':      img_after['filename'] + '.tif',
            })

    matches_df = pd.DataFrame(matches)
    print(f"Found {len(matches_df)} complete triplet matches out of {len(masks_overview)} events.")
    return matches_df


def find_nearest_match(
    target_dates: pd.Series,
    candidate_df: pd.DataFrame,
    tolerance_days: int = 3,
    date_col: str = 'date',
) -> pd.Series:
    """
    For each date in target_dates, find the nearest date in candidate_df
    within ±tolerance_days.  Returns a Series of matched filenames
    (NaN where no match within tolerance).

    Useful for matching Sentinel-1 acquisitions to existing S2 dates.
    """
    results = []
    for target in target_dates:
        window = candidate_df[
            (candidate_df[date_col] >= target - timedelta(days=tolerance_days)) &
            (candidate_df[date_col] <= target + timedelta(days=tolerance_days))
        ]
        if len(window) == 0:
            results.append(None)
        else:
            closest = window.iloc[(window[date_col] - target).abs().argsort().iloc[0]]
            results.append(closest['filename'] + '.tif')
    return pd.Series(results, index=target_dates.index)
