import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from scipy import stats

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

# Read the GOCE data file into dataframe
df_goce = pd.read_csv(
    "goce_denswind_ac082_v2_0_2013-06.txt", sep="\s+", comment="#", header=None
)

# Name the columns
df_goce.columns = [
    "date",
    "time",
    "time_scale",
    "altitude",
    "longitude",
    "latitude",
    "local_solar_time",
    "argument_latitude",
    "density",
    "crosswind_east",
    "crosswind_north",
    "crosswind_up",
    "density_error",
    "crosswind_error",
    "data_flag",
    "eclipse_flag",
    "ascending_flag",
    "thruster_flag",
]

# Combine date and time into single string and create new column for them
df_goce["datetime"] = df_goce["date"] + " " + df_goce["time"]

# Convert date and time to datetime object to enable filtering of data based thereupon
df_goce["datetime"] = pd.to_datetime(df_goce["datetime"])

# Drop ununsed columns
df_goce = df_goce.drop(
    columns=[
        "date",
        "time",
        "time_scale",
        "crosswind_east",
        "crosswind_north",
        "crosswind_up",
        "crosswind_error",
        "data_flag",
        "eclipse_flag",
        "ascending_flag",
        "thruster_flag",
    ]
)

# Isolate 2 weeks centred on our date of interest: 15th June 2013
start_day = 2
stop_day = 30
month = 6
year = 2013
start_date = datetime.datetime(year, month, start_day)
stop_date = datetime.datetime(year, month, stop_day)
df_goce = df_goce[
    (df_goce["datetime"] >= start_date)
    & (df_goce["datetime"] < stop_date)
]

# Reduce the dataset by only keeping every N-th sample
# reduces the number of API calls
reduction_factor = 500
df_goce = df_goce.iloc[::reduction_factor, :]

# Create geomagnetic indices lookup dataframe for the month

# Fetch geomagnetic indices from local file for the same year and month
# One or more space is considered a separator
df_Kp = pd.read_csv("kp1306.tab", sep="\s+", comment="#", header=None)
#
df_Kp.columns = [
    "date",
    "kp0",
    "kp1",
    "kp2",
    "kp3",
    "kp4",
    "kp5",
    "kp6",
    "kp7",
    "kp_sum",
    "quiet_disturbed_day",
    "Ap",
    "Cp",
]

# Drop unused columns
df_Kp = df_Kp.drop(
    columns=[
        "kp0",
        "kp1",
        "kp2",
        "kp3",
        "kp4",
        "kp5",
        "kp6",
        "kp7",
        "kp_sum",
        "quiet_disturbed_day",
        "Cp",
    ]
)

# Convert date to datetime object
# Will be used to look up geomag index on date of measurement
df_Kp["date"] = pd.to_datetime(df_Kp["date"], format="%y%m%d")

# Look up the geomag index at each date of measurement
df_goce["Ap"] = [
    df_Kp["Ap"][df_Kp["date"].searchsorted(dt) - 1] for dt in df_goce.datetime.values
]

# Create radio flux lookup dataframe for the month

# Fetch radio flux data for this year from local file
df_f107 = pd.read_csv("2013_DSD.txt", sep="\s+", comment="#", header=None)

df_f107.columns = [
    "year",
    "month",
    "day",
    "radio_flux",
    "sunspot_number",
    "sunspot_area",
    "new_regions",
    "solar_mean_field",
    "goes_xray_flux",
    "flares_c",
    "flares_m",
    "flares_x",
    "flares_s",
    "flares_1",
    "flares_2",
    "flares_3",
]

# Create new column of type datetime based on the ymd columns
# Will be used to lookup radio flux on date of measurement
df_f107["date"] = pd.to_datetime(df_f107[["year", "month", "day"]])

# Drop unused columns
df_f107 = df_f107.drop(
    columns=[
        "year",
        "month",
        "day",
        "sunspot_number",
        "sunspot_area",
        "new_regions",
        "solar_mean_field",
        "goes_xray_flux",
        "flares_c",
        "flares_m",
        "flares_x",
        "flares_s",
        "flares_1",
        "flares_2",
        "flares_3",
    ]
)

# Look up the radio flux for the day before each date of measurement
# Create new column in the original dataframe
result = []
for dt in df_goce.datetime.values:
    # find index of row in radio flux, will be day after, so subtract 1
    index = df_f107["date"].searchsorted(dt) - 1
    # we want to day before
    result.append(df_f107["radio_flux"][index - 1])
df_goce["f107"] = result

# Iterate ove the radio flux data and calculate 81 day averages
# if there are enough datapoints either side, otherwise assign zero
avg_flux_vals = []
for i, flux in enumerate(df_f107["radio_flux"].values):
    #
    if 40 < i < len(df_f107["radio_flux"].values) - 40:
        avg_flux_vals.append(np.mean(df_f107["radio_flux"].values[i - 40 : i + 40]))
    else:
        avg_flux_vals.append(0)

df_f107["radio_flux_avg"] = avg_flux_vals


# Look up the 81 day average radio flux at each date of measurement
df_goce["f107a"] = [
    df_f107["radio_flux_avg"][df_f107["date"].searchsorted(dt) - 1]
    for dt in df_goce.datetime.values
]

def fetch_density_from_api(row, url):
    """
    Make an API call to sample the atmospheric density using the 
    NRLMSISE-00 model. 
    
    Args: row of pandas dataframe containing conditions at time of measurement
    Returns:
        density in g/cm3
    
    """
    # Hit the Amentum Atmosphere API to calculate total mass density
    # according to NRLMSISE-00 and JB2008 endpoints
    # params common to both

    payload = {
        "altitude": row["altitude"] / 1000.0,  # convert to kms
        "geodetic_latitude": row["latitude"],
        "geodetic_longitude": row["longitude"],
        "year" : row["datetime"].year,
        "month" : row["datetime"].month,
        "day" : row["datetime"].day,
        "utc": row["datetime"].hour
        + row["datetime"].minute / 60,  # TODO this must be decimal
    }
    # additional required for the nrl one
    if "nrlmsise00" in url:
        payload.update({
            "f107a": row["f107a"],
            "f107": row["f107"],
            "ap": row["Ap"]
        })

    # Boom, hit it! Then return the JSONs
    try:
        response = requests.get(url, params=payload)
    except requests.exceptions.RequestException as e:
        print(e)
        raise KeyboardInterrupt
    else:
        return response.json()

# limits for binning of timestamp and arg of lat

# NOTE this was optimised to ensure sufficient bin widths such that at least a data 
# point per bin with sufficiently sparse GOCE data to ensure daily API quote not exceeded 
# for users. Resolution can be improved for staging and on-premises deployments. 
time_delta_low = 0
time_delta_high = (stop_date - start_date).total_seconds()

# go for daily bins. 
seconds_per_hour = 60 * 60 
seconds_per_day = 60 * 60 * 24 
# bin to ensure final edge is considered
tds = np.arange(time_delta_low, time_delta_high+seconds_per_day*2, seconds_per_day*2)

arg_lat_delta = 10 # degree
arg_lats = np.arange(0,360+arg_lat_delta,arg_lat_delta)

# Convert datetimes to delta since first measurements
# This will be used for the binning and plotting
# (avoids using datetime objects)
time_deltas = df_goce["datetime"].values - df_goce["datetime"].values.min()

# Convert time_deltas to seconds, will also convert to float type
time_deltas = [t / np.timedelta64(1, "s") for t in time_deltas]

# Calculate the densities as mean values lying within 2d grid of bins
densities = stats.binned_statistic_2d(
    time_deltas,
    df_goce["argument_latitude"].values,
    df_goce["density"].values,
    statistic="mean",
    bins=(tds, arg_lats),
)

# Calculate NRLMSISE-00 model densities using the API

for endpoint in ["nrlmsise00", "jb2008"]:

    url = args.hostname + "/api/" + endpoint

    print("WARNING: {} requests to Amentum API, may exceed quota".format(len(df_goce)))

    # Apply the function call onto each row of the dataframe
    res = df_goce.apply(fetch_density_from_api, args=(url,), axis=1)

    # Convert from g/cm3 to kg/m3
    df_goce[endpoint] = [row["total_mass"]["value"] * 1e-3 * 1e6 for row in res.values]

    # Prepare data for plotting
    densities_api = stats.binned_statistic_2d(
        time_deltas,
        df_goce["argument_latitude"].values,
        df_goce[endpoint].values,
        statistic="mean",
        bins=(tds, arg_lats),
    )

    fig_cont, (ax_goce, ax_api) = plt.subplots(nrows=2, sharex=True)
    fig_cont.suptitle("GOCE (top) vs "+endpoint.upper()+" (bottom)")

    cs0 = ax_goce.imshow(
        densities.statistic.T,
        extent=(start_day, stop_day, arg_lats.min(), arg_lats.max()),
        origin="lower",
        aspect="auto",
        cmap=plt.cm.jet,
        vmin=df_goce["density"].values.min(),
        vmax=df_goce["density"].values.max(),
    )

    cs1 = ax_api.imshow(
        densities_api.statistic.T,
        extent=(start_day, stop_day, arg_lats.min(), arg_lats.max()),
        origin="lower",
        aspect="auto",
        cmap=plt.cm.jet,
        vmin=df_goce["density"].values.min(),
        vmax=df_goce["density"].values.max(),
    )

    for ax in [ax_goce, ax_api]:
        # Fetch the labels for the api sourced data
        ax.set_ylabel("Argument of Latitude, deg")
        ax.set_yticks(np.arange(0, 360, 90))
        ax.set_xticks(np.arange(start_day, stop_day, 5))

    # Set x labels on bottom plot only
    ax_api.set_xlabel(datetime.date(year, month, 1).strftime("%B %Y"))

    # Format colorbar axis
    cb = fig_cont.colorbar(cs1, ax=list((ax_goce, ax_api)), format="%3.1e")

    cb.set_label("Density " + r"$kgm^{-3}$")

    fig_cont.savefig("Density_GOCE_vs_{}.png".format(endpoint))

    # Now plot the profiles for a particular argument latitude

    # TODO functionise this to plot for different arg of lats

    fig_prof = plt.figure()
    ax_prof = fig_prof.add_subplot(111)
    ax_prof.set_xlabel(datetime.date(year, month, 1).strftime("%B %Y"))
    ax_prof.set_ylabel("Density " + r"$kgm^{-3}$")

    midlat_index = np.searchsorted(arg_lats, 180)

    arg_lat_of_interest = arg_lats[midlat_index]

    ax_prof.plot(tds[:-1], densities.statistic.T[midlat_index, :], label="GOCE")

    ax_prof.plot(
        tds[:-1], densities_api.statistic.T[midlat_index, :], label=endpoint.upper()
    )

    labels = [item.get_text() for item in ax_prof.get_xticklabels()]


    def format_func(value, tick_number):
        """
        Function to convert tick labels from seconds elapsed to 
        day of date.
        
        """
        return start_day + int(value / seconds_per_day)


    ax_prof.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax_prof.legend()

    fig_prof.suptitle(
        "Argument of latitude {0:.2f} deg".format(arg_lat_of_interest), fontsize=12
    )

    fig_prof.savefig("Density_vs_day_AOL_{}_{}.png".format(
        int(arg_lat_of_interest), endpoint.upper()))
