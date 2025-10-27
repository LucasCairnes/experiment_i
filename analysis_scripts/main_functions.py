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

def remove_offsets(data_df, stationary_start):
    raw_cols = ["ax (m/s^2)", "ay (m/s^2)", "az (m/s^2)", "gx (deg/s)", "gy (deg/s)", "gz (deg/s)"]

    to_clean = data_df.copy()

    if stationary_start:
        mask = to_clean["t (s)"] < stationary_start
        for col in raw_cols:
            offset = to_clean.loc[mask, col].mean()
            to_clean[col] -= offset

    return to_clean

def filter_gyroscopes(data_df, low_passes, high_passes, filter_order=2):
        
    time = data_df['t (s)']
    N = len(time)
    sampling_rate = 1 / np.mean(np.diff(time))
    
    gyro_cols = ["gx (deg/s)", "gy (deg/s)", "gz (deg/s)"]

    for gyro, high, low in zip(gyro_cols, high_passes, low_passes):
        to_filter = np.array(data_df[gyro])
        nyquist_freq = 0.5 * sampling_rate

        b, a = butter(filter_order, [high / nyquist_freq, low / nyquist_freq], btype='band')

        filtered_data = filtfilt(b, a, to_filter)

        data_df[gyro] = filtered_data
    
    return data_df

def process_orientation(data_df, stationary_start, stationary_end):
    
    df_out = data_df.copy()

    time = df_out['t (s)'].values
    
    stationary_mask = (time > stationary_start) & (time < stationary_end)
    
    gx_rad = np.radians(df_out['gx (deg/s)'].values)
    gy_rad = np.radians(df_out['gy (deg/s)'].values)
    gz_rad = np.radians(df_out['gz (deg/s)'].values)
    
    roll = scipy.integrate.cumulative_trapezoid(gx_rad, time, initial=0)
    pitch = scipy.integrate.cumulative_trapezoid(gy_rad, time, initial=0)
    yaw = scipy.integrate.cumulative_trapezoid(gz_rad, time, initial=0)
            
    roll = detrend(roll)
    pitch = detrend(pitch)
    yaw = detrend(yaw)

    roll[~stationary_mask] = 0.0
    pitch[~stationary_mask] = 0.0
    yaw[~stationary_mask] = 0.0

    df_out['roll (deg)'] = np.degrees(roll)
    df_out['pitch (deg)'] = np.degrees(pitch)
    df_out['yaw (deg)'] = np.degrees(yaw)

    return df_out

def apply_low_pass(data_df, low_passes, filter_order=2):

    filtered_df = data_df.copy()

    time = filtered_df['t (s)']
    sampling_rate = 1 / np.mean(np.diff(time))
    nyquist_freq = 0.5 * sampling_rate

    raw_cols = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']

    for col, low_pass in zip(raw_cols, low_passes):
        raw_data = filtered_df[col]
        b, a = butter(filter_order, low_pass / nyquist_freq, btype='low')
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

def remove_gravity(data_df, g=9.80665):
    df_out = data_df.copy()

    ax = df_out['ax (m/s^2)'].values
    ay = df_out['ay (m/s^2)'].values
    az = df_out['az (m/s^2)'].values

    roll_rad = np.radians(df_out['roll (deg)'].values)
    pitch_rad = np.radians(df_out['pitch (deg)'].values)

    g_x = -g * np.sin(pitch_rad)
    g_y = g * np.sin(roll_rad) * np.cos(pitch_rad)
    g_z = g * np.cos(roll_rad) * np.cos(pitch_rad)

    df_out['ax (m/s^2)'] = ax - g_x
    df_out['ay (m/s^2)'] = ay - g_y
    df_out['az (m/s^2)'] = az - g_z

    return df_out

def rotate_acceleration(data_df):
    df_out = data_df.copy()

    ax_linear = df_out['ax (m/s^2)'].values
    ay_linear = df_out['ay (m/s^2)'].values
    az_linear = df_out['az (m/s^2)'].values

    phi = np.radians(df_out['roll (deg)'].values)
    theta = np.radians(df_out['pitch (deg)'].values)
    psi = np.radians(df_out['yaw (deg)'].values)

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    ax_world = (c_psi * c_theta) * ax_linear + \
               (c_psi * s_theta * s_phi - s_psi * c_phi) * ay_linear + \
               (c_psi * s_theta * c_phi + s_psi * s_phi) * az_linear
    
    ay_world = (s_psi * c_theta) * ax_linear + \
               (s_psi * s_theta * s_phi + c_psi * c_phi) * ay_linear + \
               (s_psi * s_theta * c_phi - c_psi * s_phi) * az_linear

    az_world = (-s_theta) * ax_linear + \
               (c_theta * s_phi) * ay_linear + \
               (c_theta * c_phi) * az_linear
    
    df_out['ax_world (m/s^2)'] = ax_world
    df_out['ay_world (m/s^2)'] = ay_world
    df_out['az_world (m/s^2)'] = az_world

    return df_out

def integrate_acceleration(data_df, stationary_start, stationary_end):
    
    df_integrated = data_df.copy()
    time_col = df_integrated['t (s)'].values

    motion_mask = (time_col >= stationary_start) & (time_col <= stationary_end)

    ax_vals = df_integrated['ax_world (m/s^2)'].values
    ay_vals = df_integrated['ay_world (m/s^2)'].values
    az_vals = df_integrated['az_world (m/s^2)'].values

    vel_x_raw = scipy.integrate.cumulative_trapezoid(ax_vals, time_col, initial=0)
    vel_y_raw = scipy.integrate.cumulative_trapezoid(ay_vals, time_col, initial=0)
    vel_z_raw = scipy.integrate.cumulative_trapezoid(az_vals, time_col, initial=0)
        
    vel_x_final = np.zeros(len(time_col))
    vel_y_final = np.zeros(len(time_col))
    vel_z_final = np.zeros(len(time_col))

    vel_x_segment = vel_x_raw[motion_mask]
    vel_x_final[motion_mask] = detrend(vel_x_segment)
    
    vel_y_segment = vel_y_raw[motion_mask]
    vel_y_final[motion_mask] = detrend(vel_y_segment)

    vel_z_segment = vel_z_raw[motion_mask]
    vel_z_final[motion_mask] = detrend(vel_z_segment)
    
    df_integrated['vel_x (m/s)'] = vel_x_final
    df_integrated['vel_y (m/s)'] = vel_y_final
    df_integrated['vel_z (m/s)'] = vel_z_final

    return df_integrated

def apply_high_pass(data_df, filter_order, passes):
    df_out = data_df.copy()

    time = df_out['t (s)']
    sampling_rate = 1 / np.mean(np.diff(time))
    nyquist_freq = 0.5 * sampling_rate

    cols = ['ax_world (m/s^2)', 'ay_world (m/s^2)', 'az_world (m/s^2)']

    for col, high_pass in zip(cols, passes):
        unfiltered_col = df_out[col]
        b, a = butter(filter_order, high_pass / nyquist_freq, btype='high')
        filtered_data = filtfilt(b, a, unfiltered_col)
        df_out[col] = filtered_data

    return df_out

def integrate_velocity(data_df, stationary_start, stationary_end):
    
    df_integrated = data_df.copy()
    time_col = df_integrated['t (s)'].values

    motion_mask = (time_col >= stationary_start) & (time_col <= stationary_end)

    pos_x_raw = scipy.integrate.cumulative_trapezoid(df_integrated["vel_x (m/s)"], time_col, initial=0)
    pos_y_raw = scipy.integrate.cumulative_trapezoid(df_integrated["vel_y (m/s)"], time_col, initial=0)
    pos_z_raw = scipy.integrate.cumulative_trapezoid(df_integrated["vel_z (m/s)"], time_col, initial=0)

    pos_x_final = np.zeros(len(time_col))
    pos_y_final = np.zeros(len(time_col))
    pos_z_final = np.zeros(len(time_col))

    pos_x_segment = pos_x_raw[motion_mask]
    pos_x_final[motion_mask] = detrend(pos_x_segment)

    pos_y_segment = pos_y_raw[motion_mask]
    pos_y_final[motion_mask] = detrend(pos_y_segment)

    pos_z_segment = pos_z_raw[motion_mask]
    pos_z_final[motion_mask] = detrend(pos_z_segment)

    df_integrated['d_x (m)'] = pos_x_final
    df_integrated['d_y (m)'] = pos_y_final
    df_integrated['d_z (m)'] = pos_z_final

    return df_integrated

def plot_cols(data_df, y_axis, x_axis="t"):
    
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