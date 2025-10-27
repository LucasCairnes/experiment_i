import pandas as pd
from main_functions import (
    remove_offsets, apply_savgol,
    process_orientation, remove_gravity, rotate_acceleration, 
    integrate_acceleration, integrate_velocity,
    plot_fft, plot_cols, col_dict
)

excel_data = pd.read_excel("raw_data/shapes/21_10_medium_square.xlsx")

rename_map = {key: val for key, val in col_dict.items() if key in excel_data.columns}
excel_data = excel_data.rename(columns=rename_map)

plot_fft(excel_data, selected_axis="ay")

STATIONARY_START = 13.5
STATIONARY_END = 19.3

cleaned_data = remove_offsets(
    excel_data,
    stationary_start=STATIONARY_START
)

savgol_window = 21
savgol_order = 3

smoothed_data = cleaned_data.copy()
accel_keys_to_smooth = ['ax', 'ay', 'az']
gyro_keys_to_smooth = ['gx', 'gy', 'gz']

for key in accel_keys_to_smooth + gyro_keys_to_smooth:
    smoothed_data = apply_savgol(
        smoothed_data, 
        filter_col_name=key,
        window=savgol_window, 
        order=savgol_order
    )

oriented = process_orientation(
    smoothed_data,
    stationary_start=STATIONARY_START,
    stationary_end=STATIONARY_END
)

gravity_removed = remove_gravity(
    oriented
)

rotated = rotate_acceleration(
    gravity_removed
)

integrated_accel = integrate_acceleration(
    rotated,
    stationary_start=STATIONARY_START,
    stationary_end=STATIONARY_END
)

integrated_df = integrate_velocity(
    integrated_accel,
    stationary_start=STATIONARY_START,
    stationary_end=STATIONARY_END
)

moving_mask = (integrated_df["t (s)"] > STATIONARY_START) & (integrated_df["t (s)"] < STATIONARY_END)

plot_cols(integrated_df[moving_mask], x_axis="t", y_axis="d_x")