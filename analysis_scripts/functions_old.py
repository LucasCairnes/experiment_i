import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, detrend, savgol_filter
import scipy.integrate
import math

col_dict = {"t": 't (s)',
            "ax": 'ax (m/s^2)',
            "ay": 'ay (m/s^2)',
            "az": 'az (m/s^2)',
            "gx": 'gx (deg/s)',
            "gy": 'gy (deg/s)',
            "gz": 'gz (deg/s)',
            "vel_x": 'vel_x (m/s)',
            "vel_y": 'vel_y (m/s)',
            "vel_z": 'vel_z (m/s)',
            "d_x": 'd_x (m)',
            "d_y": 'd_y (m)',
            "d_z": 'd_z (m)',
            "theta_x": 'theta_x (deg)',
            "theta_y": 'theta_y (deg)',
            "theta_z": 'theta_z (deg)'}

raw_cols = ["ax (m/s^2)", "ay (m/s^2)", "az (m/s^2)", "gx (deg/s)", "gy (deg/s)", "gz (deg/s)"]

def clean_data(data_df, offsets=None, stationary_time=None, g_col=None):
    raw_cols = ["ax (m/s^2)", "ay (m/s^2)", "az (m/s^2)", "gx (deg/s)", "gy (deg/s)", "gz (deg/s)"]
    to_clean = data_df.copy()

    if g_col:
        gravity_val = 9.81
        if g_col in to_clean.columns:
            to_clean[g_col] -= gravity_val

    if offsets:
        offset_series = pd.Series(offsets, index=raw_cols)
        to_clean[raw_cols] = to_clean[raw_cols] - offset_series

    return to_clean

def clean_data2(data_df, stationary_start=None, g_col=None):

    raw_cols = ["ax (m/s^2)", "ay (m/s^2)", "az (m/s^2)"]
    to_clean = data_df.copy()

    if g_col:
        gravity_val = 9.81
        to_clean[col_dict[g_col]] -= gravity_val

    if stationary_start:
        mask = to_clean["t (s)"] < stationary_start
        for col in raw_cols:
            offset = to_clean.loc[mask, col].mean()
            to_clean[col] -= offset

    return to_clean

def apply_filter(data_df, filter_col, low_pass=None, high_pass=None, filter_order=2):
    
    column = col_dict[filter_col]
    
    time = data_df['t (s)']
    N = len(time)
    sampling_rate = 1 / np.mean(np.diff(time))
    
    to_filter = np.array(data_df[column])
    nyquist_freq = 0.5 * sampling_rate
    
    if low_pass and high_pass:
        b, a = butter(filter_order, [high_pass / nyquist_freq, low_pass / nyquist_freq], btype='band')
    elif low_pass:
        b, a = butter(filter_order, low_pass / nyquist_freq, btype='low')
    elif high_pass:
        b, a = butter(filter_order, high_pass / nyquist_freq, btype='high')

    filtered_data = filtfilt(b, a, to_filter)

    data_df[column] = filtered_data  
    
    return data_df

def process_orientation(data_df, alpha=0.98):
    
    df_out = data_df.copy()
    
    time = df_out['t (s)'].values
    dt = np.diff(time, prepend=time[0])
    dt[0] = 0.0
    
    ax = df_out['ax (m/s^2)'].values
    ay = df_out['ay (m/s^2)'].values
    az = df_out['az (m/s^2)'].values
    
    gx_rad = np.radians(df_out['gx (deg/s)'].values)
    gy_rad = np.radians(df_out['gy (deg/s)'].values)
    gz_rad = np.radians(df_out['gz (deg/s)'].values)
    
    N = len(df_out)
    
    roll = np.zeros(N)
    pitch = np.zeros(N)
    yaw = np.zeros(N)
    
    roll[0] = np.arctan2(ay[0], az[0])
    pitch[0] = np.arctan2(-ax[0], np.sqrt(ay[0]**2 + az[0]**2))
    yaw[0] = 0.0

    for i in range(1, N):
        
        roll_gyro = roll[i-1] + ((gx_rad[i] + gx_rad[i-1]) / 2.0) * dt[i]
        pitch_gyro = pitch[i-1] + ((gy_rad[i] + gy_rad[i-1]) / 2.0) * dt[i]
        yaw_gyro = yaw[i-1] + ((gz_rad[i] + gz_rad[i-1]) / 2.0) * dt[i]
        
        roll_accel = np.arctan2(ay[i], az[i])
        pitch_accel = np.arctan2(-ax[i], np.sqrt(ay[i]**2 + az[i]**2))
        
        roll[i] = alpha * (roll_gyro) + (1.0 - alpha) * (roll_accel)
        pitch[i] = alpha * (pitch_gyro) + (1.0 - alpha) * (pitch_accel)
        yaw[i] = yaw_gyro

    df_out['roll (deg)'] = np.degrees(roll)
    df_out['pitch (deg)'] = np.degrees(pitch)
    df_out['yaw (deg)'] = np.degrees(yaw)
    
    return df_out

def apply_filter2(data_df, accel_cutoffs, gyro_cutoffs, filter_order=2):

    filtered_df = data_df.copy()

    time = filtered_df['t (s)']
    sampling_rate = 1 / np.mean(np.diff(time))
    nyquist_freq = 0.5 * sampling_rate

    accel_columns = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']
    gyro_columns = ['gx (deg/s)', 'gy (deg/s)', 'gz (deg/s)']

    for col, (low_cutoff, high_cutoff) in zip(accel_columns, accel_cutoffs):
        raw_data = filtered_df[col]
        Wn = [low_cutoff / nyquist_freq, high_cutoff / nyquist_freq]
        b, a = butter(filter_order, Wn, btype='band')
        filtered_data = filtfilt(b, a, raw_data)
        filtered_df.loc[:, col] = filtered_data
        
    for col, (low_cutoff, high_cutoff) in zip(gyro_columns, gyro_cutoffs):
        raw_data = filtered_df[col]
        Wn = [low_cutoff / nyquist_freq, high_cutoff / nyquist_freq]
        b, a = butter(filter_order, Wn, btype='band')
        filtered_data = filtfilt(b, a, raw_data)
        filtered_df.loc[:, col] = filtered_data

    return filtered_df


def plot_fft(data_df, selected_axis='y', zoom_on_motion=False, zero_pad_factor=4):
    
    time = data_df['t (s)']
    plot_col = col_dict[selected_axis]
    movement_data_original = data_df[plot_col].to_numpy()
    N_original = len(movement_data_original)

    axis_title = f"{plot_col} - Frequency Spectrum"
    
    if zero_pad_factor > 1:
        N = int(N_original * zero_pad_factor)
        padded_data = np.zeros(N)
        padded_data[:N_original] = movement_data_original
        movement_data = padded_data
    else:
        N = N_original
        movement_data = movement_data_original
        
    sampling_rate = 1 / np.mean(np.diff(time))

    yf = fft(movement_data)
    xf = fftfreq(N, 1 / sampling_rate)[:N//2]
    yf_amplitude = 2.0/N_original * np.abs(yf[0:N//2])

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'Frequency Spectrum {selected_axis.upper()}-Axis', fontsize=16)

    ax.plot(xf, yf_amplitude)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(axis_title)
    ax.grid(True)

    if zoom_on_motion:
        ax.set_xlim(0, 1)
        
        visible_mask = (xf >= 0) & (xf <= 1)
        
        if np.any(visible_mask):
            visible_amplitudes = yf_amplitude[visible_mask]
            y_max = np.max(visible_amplitudes)
            ax.set_ylim(0, y_max * 1.1)
            
        ax.set_title(f"{axis_title} (Zoomed to 0-1 Hz)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    results = {}
    results['data'] = (xf, yf_amplitude)
    results['plot'] = (fig, ax)

    return results


def integrate_data(data_df, stationary_start, stationary_end):
    
    df_integrated = data_df.copy()
    time_col = df_integrated['t (s)'].values

    stationary_mask = (time_col < stationary_end) | (time_col > stationary_start)
                                                              
    accel_cols = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']
    vel_cols = ['vel_x (m/s)', 'vel_y (m/s)', 'vel_z (m/s)']
    pos_cols = ['d_x (m)', 'd_y (m)', 'd_z (m)']

    for ax_col, vel_col, pos_col in zip(accel_cols, vel_cols, pos_cols):
        
        raw_velocity = scipy.integrate.cumulative_trapezoid(
            df_integrated[ax_col], 
            time_col, 
            initial=0
        )
        
        detrended_velocity = detrend(raw_velocity)
        
        detrended_velocity[stationary_mask] = 0.0
        
        df_integrated[vel_col] = detrended_velocity
        
        position = scipy.integrate.cumulative_trapezoid(
            df_integrated[vel_col],
            time_col, 
            initial=0
        )
        
        df_integrated[pos_col] = position

    gyro_cols = ['gx (deg/s)', 'gy (deg/s)', 'gz (deg/s)']
    angle_cols = ['theta_x (deg)', 'theta_y (deg)', 'theta_z (deg)']
    
    for gyro_col, angle_col in zip(gyro_cols, angle_cols):
        
        raw_angle = scipy.integrate.cumulative_trapezoid(
            df_integrated[gyro_col], 
            time_col, 
            initial=0
        )
        
        detrended_angle = detrend(raw_angle)
        
        detrended_angle[stationary_mask] = 0.0
        
        df_integrated[angle_col] = detrended_angle

    return df_integrated


def plot_cols(data_df, x_axis, y_axis):
    
    if x_axis in col_dict:
        temp = col_dict[x_axis]
        x_axis = temp
    if y_axis in col_dict:
        temp = col_dict[y_axis]
        y_axis = temp

    plot_x = data_df[x_axis]
    plot_y = data_df[y_axis]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_x, plot_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} vs {x_axis}')
    plt.grid(True)
    plt.show()

def apply_savgol(data_df, filter_col):
    
    column = col_dict[filter_col]
    to_filter = data_df[column]
    
    window = 75
    order = 7

    filtered_data = savgol_filter(to_filter, window_length = int(window), polyorder = int(order), mode = 'interp')

    data_df[column] = filtered_data  

    return data_df