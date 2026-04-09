import numpy as np

# Generic sub-index calculator
def calculate_subindex(conc, breakpoints):
    for bp in breakpoints:
        Clow, Chigh, Ilow, Ihigh = bp
        if Clow <= conc <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (conc - Clow) + Ilow
    return np.nan


# Breakpoints (CPCB Approximate)
PM25_BP = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, 500, 401, 500),
]

PM10_BP = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 250, 101, 200),
    (251, 350, 201, 300),
    (351, 430, 301, 400),
    (431, 600, 401, 500),
]

NO2_BP = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 180, 101, 200),
    (181, 280, 201, 300),
    (281, 400, 301, 400),
    (401, 1000, 401, 500),
]

SO2_BP = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 380, 101, 200),
    (381, 800, 201, 300),
    (801, 1600, 301, 400),
    (1601, 2000, 401, 500),
]

CO_BP = [
    (0, 1, 0, 50),
    (1.1, 2, 51, 100),
    (2.1, 10, 101, 200),
    (10.1, 17, 201, 300),
    (17.1, 34, 301, 400),
    (34.1, 50, 401, 500),
]

O3_BP = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 168, 101, 200),
    (169, 208, 201, 300),
    (209, 748, 301, 400),
    (749, 1000, 401, 500),
]

NH3_BP = [
    (0, 200, 0, 50),
    (201, 400, 51, 100),
    (401, 800, 101, 200),
    (801, 1200, 201, 300),
    (1201, 1800, 301, 400),
    (1801, 2000, 401, 500),
]


def compute_aqi(row):
    # 🔥 Normalize keys
    row = row.copy()
    row.index = [col.lower() for col in row.index]

    sub_indices = []

    if not np.isnan(row.get("pm25", np.nan)):
        sub_indices.append(calculate_subindex(row["pm25"], PM25_BP))

    if not np.isnan(row.get("pm10", np.nan)):
        sub_indices.append(calculate_subindex(row["pm10"], PM10_BP))

    if not np.isnan(row.get("no2", np.nan)):
        sub_indices.append(calculate_subindex(row["no2"], NO2_BP))

    if not np.isnan(row.get("so2", np.nan)):
        sub_indices.append(calculate_subindex(row["so2"], SO2_BP))

    if not np.isnan(row.get("co", np.nan)):
        sub_indices.append(calculate_subindex(row["co"], CO_BP))

    if not np.isnan(row.get("o3", np.nan)):
        sub_indices.append(calculate_subindex(row["o3"], O3_BP))

    if not np.isnan(row.get("nh3", np.nan)):
        sub_indices.append(calculate_subindex(row["nh3"], NH3_BP))

    if len(sub_indices) == 0:
        return np.nan

    return max(sub_indices)