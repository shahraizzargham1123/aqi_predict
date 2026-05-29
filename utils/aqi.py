"""
US EPA Air Quality Index calculation.

Open-Meteo hands us raw pollutant concentrations (mostly in µg/m³), but the AQI
number people actually recognise is the EPA index from 0-500. This module turns
those concentrations into that index.

A note on honesty: the official EPA AQI uses averaging windows (24h for
particulates, 8h for ozone and CO, 1h for the gases). We're working with hourly
readings and compute the sub-index straight off each hourly value. That's a
common simplification for a forecasting feature and keeps the pipeline simple,
but it's worth knowing it's not the exact regulatory number.
"""

# Each table maps a concentration band to an AQI band:
#   (concentration_low, concentration_high, aqi_low, aqi_high)
# The pollutant value has to be in the unit noted next to each table.

# PM2.5 in µg/m³
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

# PM10 in µg/m³
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

# Ozone in ppb (8-hour basis)
O3_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]

# Carbon monoxide in ppm
CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

# Sulphur dioxide in ppb
SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]

# Nitrogen dioxide in ppb
NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

# Multiplier to turn µg/m³ into ppb at 25°C / 1 atm. Works out to
# 24.45 / molecular_weight. CO we then divide by 1000 to land on ppm.
UGM3_TO_PPB = {
    "ozone": 24.45 / 48.00,
    "nitrogen_dioxide": 24.45 / 46.01,
    "sulphur_dioxide": 24.45 / 64.07,
    "carbon_monoxide": 24.45 / 28.01,
}


def _sub_index(concentration, breakpoints):
    """Linear interpolation of one pollutant onto the AQI scale."""
    if concentration is None or concentration != concentration:  # None or NaN
        return None
    # Anything above the top of the table just pins to the worst band.
    if concentration > breakpoints[-1][1]:
        return float(breakpoints[-1][3])
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return (i_high - i_low) / (c_high - c_low) * (concentration - c_low) + i_low
    return None


def compute_aqi(pm2_5=None, pm10=None, ozone=None, nitrogen_dioxide=None,
                sulphur_dioxide=None, carbon_monoxide=None):
    """
    Overall AQI for a single reading. The EPA index is the worst sub-index
    across all the pollutants we have data for, so we compute each one we can
    and take the max.
    """
    sub_indices = []

    if pm2_5 is not None:
        sub_indices.append(_sub_index(pm2_5, PM25_BREAKPOINTS))
    if pm10 is not None:
        sub_indices.append(_sub_index(pm10, PM10_BREAKPOINTS))
    if ozone is not None:
        ppb = ozone * UGM3_TO_PPB["ozone"]
        sub_indices.append(_sub_index(ppb, O3_BREAKPOINTS))
    if nitrogen_dioxide is not None:
        ppb = nitrogen_dioxide * UGM3_TO_PPB["nitrogen_dioxide"]
        sub_indices.append(_sub_index(ppb, NO2_BREAKPOINTS))
    if sulphur_dioxide is not None:
        ppb = sulphur_dioxide * UGM3_TO_PPB["sulphur_dioxide"]
        sub_indices.append(_sub_index(ppb, SO2_BREAKPOINTS))
    if carbon_monoxide is not None:
        ppm = carbon_monoxide * UGM3_TO_PPB["carbon_monoxide"] / 1000.0
        sub_indices.append(_sub_index(ppm, CO_BREAKPOINTS))

    sub_indices = [s for s in sub_indices if s is not None]
    if not sub_indices:
        return None
    return round(max(sub_indices))


def aqi_category(aqi):
    """Human-friendly label for an AQI value, matching the EPA colour bands."""
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"
