import pandas as pd
import numpy as np

radius_cm = 10
period_s = 10
sampling_rate_hz = 100
std_dev = 0


radius_m = radius_cm / 100.0 

angular_velocity_rad_s = 2 * np.pi / period_s

acceleration_magnitude = (angular_velocity_rad_s ** 2) * radius_m

num_samples = int(period_s * sampling_rate_hz)
time_s = np.linspace(0, period_s, num_samples, endpoint=False)

angle_rad = angular_velocity_rad_s * time_s

ax_ideal = -acceleration_magnitude * np.cos(angle_rad)
ay_ideal = -acceleration_magnitude * np.sin(angle_rad)

az_ideal = np.zeros(num_samples)

noise_ax = np.random.normal(0, std_dev, num_samples)
noise_ay = np.random.normal(0, std_dev, num_samples)
noise_az = np.random.normal(0, std_dev, num_samples)

ax_noisy = ax_ideal + noise_ax
ay_noisy = ay_ideal + noise_ay
az_noisy = az_ideal + noise_az

gx = np.zeros(num_samples)
gy = np.zeros(num_samples)
gz = np.zeros(num_samples)

data = {
    't (s)': time_s,
    'ax (m/s^2)': ax_noisy,
    'ay (m/s^2)': ay_noisy,
    'az (m/s^2)': az_noisy,
    'gx (deg/s)': gx,
    'gy (deg/s)': gy,
    'gz (deg/s)': gz
}
df = pd.DataFrame(data)

file_path = f"GY-521_tests/artifical_data/no_noise.xlsx"

df.to_excel(file_path)


