import pandas as pd
from main_functions import remove_offsets, apply_low_pass, filter_gyroscopes, integrate_acceleration, plot_fft, plot_cols, process_orientation, remove_gravity, rotate_acceleration, apply_high_pass, integrate_velocity

excel_data = pd.read_excel("GY-521_tests/Slow Square 2.xlsx")
raw_cols = ['ax', 'ay', 'az']
gyro_cols = ["gx", "gy", "gz"]

low_passes = [25, 25, 2]
high_passes = [0.05, 0.06, 1]

gyro_high = [1, 1, 1]
gyro_low = [5, 5, 5] 

STATIONARY_START = 5.38
STATIONARY_END = 14.26

cleaned_data = remove_offsets(
    excel_data,
    stationary_start=STATIONARY_START
)

low_passed = apply_low_pass(
    cleaned_data,
    low_passes=low_passes,
    filter_order=4
)

filtered_gyro = filter_gyroscopes(
    low_passed,
    low_passes=gyro_low,
    high_passes=gyro_high
)

oriented = process_orientation(
    filtered_gyro,
    stationary_start=STATIONARY_START,
    stationary_end=STATIONARY_END
)

gravity_removed = remove_gravity(
    oriented
)

rotated = rotate_acceleration(
    gravity_removed
)

high_passed = apply_high_pass(
    rotated,
    filter_order=2,
    passes=high_passes
)

integrated_accel = integrate_acceleration(
    high_passed,
    stationary_start=STATIONARY_START,
    stationary_end=STATIONARY_END
)

integrated_df = integrate_velocity(
    integrated_accel,
    stationary_start=STATIONARY_START,
    stationary_end=STATIONARY_END
)

stationary_mask = (integrated_df["t (s)"] > 5.38) & (integrated_df["t (s)"] < 14.26)

plot_cols(integrated_df, x_axis="d_x", y_axis="d_y")