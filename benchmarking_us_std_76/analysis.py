import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

# Define constants

# gravitational constant at sea level
g_0 = 9.80665  # [m/s2]
# Molar mass of dry air
M_da = 28.96442 * 1e-3  # [kg/mol]
# Universal molar ga constant Rg
R_g = 8.31432  # [J/mol/K]
# radius of the earth
r_0 = 6356.776 * 1e3  # m

def CalcGeopotentialHeight(Z):
    """
    Calculate geoptential height from geometric height

    Args - geometric height in m as float

    Returns - geopotential height in m as float 

    """
    return (r_0 * Z) / (r_0 + Z)

def CalcTemp(delta_H, T_0, Lapse):
    """
    Calculates atmospheric temperature assuming constant lapse rate
    
    Args
    
    delta_H - difference in geopotential heights between layers in m
    T_0 - base temperature of previous layer in K
    Lapse - lapse rate in K/m 
    
    Return 
    
    temperature - float in units of K
    
    """

    return T_0 + Lapse * delta_H


# Function to calculate pressure
def CalcPressure(delta_H, T_0, P_0, Lapse):
    """
    Calculates the atmospheric pressure 
        
    Args
    
    delta_H - difference in geopotential heights between layers in m
    T_0 - base temperature of previous layer in K
    P_0 - base pressure of previous layer in Pa
    Lapse - lapse rate in K/m 
    
    Return 
    
    pressure as float (units of Pa)
    
    """
    result = None

    try:
        if Lapse == 0:
            # We have exponential solution to the equation
            result = P_0 * np.exp(-(g_0 * M_da * delta_H) / (R_g * T_0))
        else:
            result = P_0 * np.power(
                (1 + Lapse * delta_H / T_0), -(g_0 * M_da) / (R_g * Lapse)
            )

    except RuntimeError as err:
        print("Error in pressure calculation:", err)

    return result

# Define layers 

# List of layers as dictionaries containing
# Temp, pressure, geometric and geopotential altitude at base of each
# as well as lapse rate
# all quantities in SI units
layers = []
troposphere = {
    "name": "troposphere",
    "pressure": 101.325 * 1e3,  # [Pa]
    "temp": 288.15,  # [K],
    "altitude": 0.0,  # [m]
    "geopotential": 0.0,  # [m]
    "lapse": -6.5 * 1e-3,  # [K/m]
}
layers.append(troposphere)

tropopause = {
    "name": "tropopause",
    "altitude": 11.0 * 1e3,  # [m]
    "lapse": 0.0,  # [K/m]
}
tropopause["geopotential"] = CalcGeopotentialHeight(tropopause["altitude"])  # [m]
tropopause["temp"] = CalcTemp(
    tropopause["geopotential"] - troposphere["geopotential"],
    troposphere["temp"],
    troposphere["lapse"],
)
tropopause["pressure"] = CalcPressure(
    tropopause["geopotential"] - troposphere["geopotential"],
    troposphere["temp"],
    troposphere["pressure"],
    troposphere["lapse"],
)
layers.append(tropopause)

mid_stratosphere = {
    "name": "mid_stratosphere",
    "altitude": 20.0 * 1e3,  # [m]
    "lapse": 1.0 * 1e-3,  # [K/m]
}
mid_stratosphere["geopotential"] = CalcGeopotentialHeight(
    mid_stratosphere["altitude"]
)  # [m]
mid_stratosphere["temp"] = tropopause["temp"]  # no change in previous layer
mid_stratosphere["pressure"] = CalcPressure(
    mid_stratosphere["geopotential"] - tropopause["geopotential"],
    tropopause["temp"],
    tropopause["pressure"],
    tropopause["lapse"],
)
layers.append(mid_stratosphere)

upper_stratosphere = {
    "name": "upper_stratosphere",
    "altitude": 32.0 * 1e3,  # [m]
    "lapse": 2.8 * 1e-3,  # [K/m]
}
upper_stratosphere["geopotential"] = CalcGeopotentialHeight(
    upper_stratosphere["altitude"]
)  # [m]
upper_stratosphere["temp"] = CalcTemp(
    upper_stratosphere["geopotential"] - mid_stratosphere["geopotential"],
    mid_stratosphere["temp"],
    mid_stratosphere["lapse"],
)
upper_stratosphere["pressure"] = CalcPressure(
    upper_stratosphere["geopotential"] - mid_stratosphere["geopotential"],
    mid_stratosphere["temp"],
    mid_stratosphere["pressure"],
    mid_stratosphere["lapse"],
)
layers.append(upper_stratosphere)

stratopause = {
    "name": "stratopause",
    "altitude": 47.0 * 1e3,  # [m]
    "lapse": 0.0 * 1e-3,  # [K/m]
}
stratopause["geopotential"] = CalcGeopotentialHeight(stratopause["altitude"])  # [m]
stratopause["temp"] = CalcTemp(
    stratopause["geopotential"] - upper_stratosphere["geopotential"],
    upper_stratosphere["temp"],
    upper_stratosphere["lapse"],
)
stratopause["pressure"] = CalcPressure(
    stratopause["geopotential"] - upper_stratosphere["geopotential"],
    upper_stratosphere["temp"],
    upper_stratosphere["pressure"],
    upper_stratosphere["lapse"],
)
layers.append(stratopause)

mid_mesosphere = {
    "name": "mid_mesosphere",
    "altitude": 51.0 * 1e3,  # [m]
    "lapse": -2.8 * 1e-3,  # [K/m]
}
mid_mesosphere["geopotential"] = CalcGeopotentialHeight(
    mid_mesosphere["altitude"]
)  # [m]
mid_mesosphere["temp"] = stratopause["temp"]  # no change in previous layer
mid_mesosphere["pressure"] = CalcPressure(
    mid_mesosphere["geopotential"] - stratopause["geopotential"],
    stratopause["temp"],
    stratopause["pressure"],
    stratopause["lapse"],
)
layers.append(mid_mesosphere)

upper_mesosphere = {
    "name": "upper_mesosphere",
    "altitude": 71.0 * 1e3,  # [m]
    "lapse": -2.0 * 1e-3,  # [K/m]
}
upper_mesosphere["geopotential"] = CalcGeopotentialHeight(
    upper_mesosphere["altitude"]
)  # [m]
upper_mesosphere["temp"] = CalcTemp(
    upper_mesosphere["geopotential"] - mid_mesosphere["geopotential"],
    mid_mesosphere["temp"],
    mid_mesosphere["lapse"],
)
upper_mesosphere["pressure"] = CalcPressure(
    upper_mesosphere["geopotential"] - mid_mesosphere["geopotential"],
    mid_mesosphere["temp"],
    mid_mesosphere["pressure"],
    mid_mesosphere["lapse"],
)
layers.append(upper_mesosphere)


def SampleUSStdAtmosphere(Z):
    """
    Calculates air temperature, pressure and density 
    assuming ideal gas law and constant lapse rates
    
    Args
    
    Z - float in units of m
    
    Returns 
    
    temp, pressure, density - tuple in units of T, Pa, kg/m3, respectively 
    
    """
    current_layer = None
    # identify which layer we're in , search in reverse order
    for layer in layers[::-1]:
        if Z >= layer["altitude"]:
            print("altitude {} is in layer {}".format(Z, layer["name"]))
            current_layer = layer
            break

    if current_layer is None:
        raise ValueError("Did not find layer for altitude {} m".format(Z))

    # calculate geopotential height different
    delta_H = CalcGeopotentialHeight(Z) - layer["geopotential"]

    # calculate temp and pressure according to layers and lapse rates
    pressure = CalcPressure(
        delta_H, layer["temp"], layer["pressure"], layer["lapse"]
    )
    temp = CalcTemp(delta_H, layer["temp"], layer["lapse"])

    # calculate density according ideal gas law
    try:
        result = temp, pressure, (M_da * pressure) / (R_g * temp)
    except ZeroDivisionError as err:
        print("Division by zero in density calc")

    return result


if __name__ == "__main__":

    # Obtain the hostname via command line argument (for on-premises deployment)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        dest="hostname",
        action="store",
        help="specify an alternative hostname for testing (e.g. on-premises server)",
        default="https://atmosphere.amentum.space",
    )
    args = parser.parse_args()


    # Calculate temperature and pressure profiles for altitudes up to 86 km
    altitudes = np.arange(0, 86.0, 1.0) * 1e3  # [m]

    # Hit the Amentum Atmosphere API to calculate total mass density according to NRLMSISE-00

    endpoint = args.hostname + "/api/nrlmsise00"

    # assume midnight at Greenwich, arbitrary date
    payload = {
        "year": 2013,
        "month": 6,
        "day": 16,
        "geodetic_latitude": 51.47879,
        "geodetic_longitude": 0,
        "utc": 0,
        "altitude": 80,
    }
    # values of radio flux, 81 day average of radio flux, and geomagnetic
    # Ap index are automatically fetched from online sources by the API

    densities_api = []
    temps_api = []

    for alt in altitudes:
        # Update altitude dict entry
        payload["altitude"] = alt / 1e3  # km
        try:
            response = requests.get(endpoint, params=payload)
        except requests.exceptions.RequestException as e:
            print(e)
        else:
            json_payload = response.json()
            densities_api.append(json_payload["total_mass_density"]["value"])  # [kg/m3]
            temps_api.append(json_payload["at_alt_temp"]["value"])  # [K]

    # Now calculate values based on the US Standard Atmosphere 1976

    # Calc quantities as list of tuples
    data_pts = np.array([SampleUSStdAtmosphere(alt) for alt in altitudes])

    # Isolate quantities as lists
    temperatures = data_pts.T[0]
    pressures = data_pts.T[1]
    densities = data_pts.T[2]
    geopotential_alts = np.array([CalcGeopotentialHeight(alt) for alt in altitudes])

    # now plot the air density profiles

    fig = plt.figure(figsize=(4.4, 5.6))
    ax = fig.add_subplot(111)
    ax.set_ylabel("Geopotential Altitude [km]")
    ax.set_xlabel(r"$\rho _{air} \quad [kgm^{-3}]$")

    ax.semilogx(
        densities,
        geopotential_alts / 1000.0,
        marker="None",
        linestyle="-",
        label="US Std",
    )
    ax.semilogx(
        densities_api,
        geopotential_alts / 1000.0,
        marker="None",
        linestyle="--",
        label="NRLMSISE-00",
    )

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylim(top=90)
    # ax.legend()
    ax.grid()
    ax.legend()
    # plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("air_density_profile.png")

    # now plot the temperature profile

    fig = plt.figure(figsize=(4.4, 5.6))
    ax = fig.add_subplot(111)
    ax.set_ylabel("Geopotential Altitude [km]")
    ax.set_xlabel("T [K]")

    ax.plot(
        temperatures,
        geopotential_alts / 1000.0,
        marker="None",
        linestyle="-",
        label="US Std",
    )

    ax.plot(
        temps_api,
        geopotential_alts / 1000.0,
        marker="None",
        linestyle="--",
        label="NRLMSISE-00",
    )

    plt.xlim(left=160)
    plt.xlim(right=300)
    plt.ylim(bottom=0)
    plt.ylim(top=90)
    ax.grid()
    ax.legend()
    # plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("air_temperature_profile.png")

    # pressure gradient

    fig = plt.figure(figsize=(4.4, 5.6))
    ax = fig.add_subplot(111)
    ax.set_ylabel("Geopotential Altitude [km]")
    ax.set_xlabel("P [Pa]")

    ax.semilogx(
        pressures,
        geopotential_alts / 1000.0,
        marker="None",
        linestyle="-",
        label="US Std",
    )

    # plt.xlim(left = 160)
    # plt.xlim(right = 300)
    plt.ylim(bottom=0)
    plt.ylim(top=90)
    ax.grid()

    # plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("air_pressure_profile.png")

