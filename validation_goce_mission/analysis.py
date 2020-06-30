import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from scipy import stats
import urllib

"""

Plotting 1 week of GOCE data for nominated year, month, day
Comparing measured atmospheric density with that predicted by NRLMSISE00 and 
JB2008 models accessed via the Amentum Aerospace API. 
Requires dedicated server or on-premises deployment, will exceed daily limit 
on the research plan.

"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hostname",
    dest="hostname",
    action="store",
    help="specify an alternative hostname for testing (e.g. on-premises server)",
    default="https://atmosphere.amentum.space",
)
parser.add_argument(
    '--start_date', 
    type=lambda s: datetime.datetime.strptime(s, '%Y%m%d'),
    dest="start_date",
    help="specify the start date for the analysis as YYYYMMDD",
    default="20130605"
)
parser.add_argument(
    '--goce_dir',
    dest="goce_dir",
    action="store",
    help="specify path to directory containing extracted goce data archive",
    default="./"
)

args = parser.parse_args()

mission_data_start = datetime.datetime(year=2009, month=12, day=1) 
mission_data_stop = datetime.datetime(year=2013, month=9, day=1)

if not (mission_data_start < args.start_date < mission_data_stop):
    raise ValueError(
        "Date {} is outside mission date range {} to {}".format(
        args.start_date, mission_data_start, mission_data_stop)
    )

# construct filename from year and month of interest
filename = "goce_denswind_ac082_v2_0_" \
    + args.start_date.strftime("%Y")+"-"\
    + args.start_date.strftime("%m")+".txt"

# Read the GOCE data file into dataframe
df_goce = pd.read_csv(
   args.goce_dir+"/"+filename, sep="\s+", comment="#", header=None
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

# Convert date and time to datetime object to enable filtering of data based thereuponst
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

# Isolate 1 week 
start_date = args.start_date
stop_date = args.start_date + datetime.timedelta(weeks=1)

df_goce = df_goce[
    (df_goce["datetime"] >= start_date)
    & (df_goce["datetime"] < stop_date)
]

""" Reduce the dataset by only keeping every N-th sample
 reduces the number of API calls, but requires coarser binning."""
reduction_factor = 10
df_goce = df_goce.iloc[::reduction_factor, :]

def fetch_density_from_api(row, url):
    """
    Make an API call to sample the atmospheric density using the 
    NRLMSISE-00 or JB2008 model
    
    Args: 
        row of pandas dataframe containing conditions at time of measurement
        url of the end point to hit
    Returns:
        density in kg/m3 as json
    
    """
    # params common to both
    payload = {
        "altitude": row["altitude"] / 1000.0,  # convert to kms
        "geodetic_latitude": row["latitude"],
        "geodetic_longitude": row["longitude"], 
        "year" : row["datetime"].year,
        "month" : row["datetime"].month,
        "day" : row["datetime"].day,
        "utc": row["datetime"].hour
        + row["datetime"].minute / 60, # decimal UTC hour
    }

    # Boom, hit it! Then return the JSONs
    try:
        response = requests.get(url, params=payload)
    except requests.exceptions.RequestException as e:
        print(e)
        raise KeyboardInterrupt
    else:
        return response.json()

# limits for binning of timestamp and arg of lat

# 
# we calculate dsitributions of mean density for discrete values of seconds from 
# the start date, and discrete values of argument of latitude.  
# the resolution can be tuned to reduce number of API calls for the study
time_delta_low = 0
time_delta_high = (stop_date - start_date).total_seconds()

# go for hourly or daily bins. 
seconds_per_day = 60 * 60 * 24 
seconds_per_bin = seconds_per_day * 0.1
# binning ensures the final edge is considered
tds = np.arange(time_delta_low, time_delta_high+seconds_per_bin, seconds_per_bin)

arg_lat_delta = 9 # argument of latitude resolution in degrees
arg_lats = np.arange(0,360+arg_lat_delta,arg_lat_delta)

# Convert datetimes to delta time in seconds since first measurements
# This will be used for the binning and plotting
time_deltas = df_goce["datetime"].values - df_goce["datetime"].values.min()

# Convert time_deltas to seconds, will also convert to float type
time_deltas = [t / np.timedelta64(1, 's') for t in time_deltas]

# Calculate the GOCE densities as mean values lying within 2d grid of bins
densities = stats.binned_statistic_2d(
    time_deltas,
    df_goce["argument_latitude"].values,
    df_goce["density"].values,
    statistic="median",
    bins=(tds, arg_lats),
)

# initialise the profile plot
fig_prof = plt.figure()
ax_prof = fig_prof.add_subplot(111)
ax_prof.set_xlabel(start_date.strftime("%B %Y"))
ax_prof.set_ylabel("Density " + r"$kgm^{-3}$")

midlat_index = np.searchsorted(arg_lats, 180)

arg_lat_of_interest = arg_lats[midlat_index]

ax_prof.plot(tds[:-1], 
    densities.statistic.T[midlat_index, :], 
    label="GOCE",
    marker="D")

labels = [item.get_text() for item in ax_prof.get_xticklabels()]

def format_func(value, tick_number):
    """
    Function to convert tick labels from seconds elapsed to 
    day of date.
    
    """
    return start_date.day + int(value / seconds_per_day)

ax_prof.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

# TODO add bin edges to title 

fig_prof.suptitle(
    "Median Density for AOL {}-{} deg".format(
        arg_lat_of_interest, arg_lats[midlat_index+1]
        ), fontsize=12
)

# Calculate NRLMSISE-00 model densities using the API

# plot all on same figure

fig_cont, (ax_goce, ax_nrlmsise00, ax_jb2008) = \
    plt.subplots(nrows=3, sharex=True, figsize=(8,12))

fig_cont.suptitle(start_date.strftime("%B %Y"))

images = []

img = ax_goce.imshow(
    densities.statistic.T,
    extent=(start_date.day, stop_date.day, arg_lats.min(), arg_lats.max()),
    origin="lower",
    aspect="auto",
    cmap=plt.cm.jet,
    vmin=df_goce["density"].values.min(),
    vmax=df_goce["density"].values.max(),
)

images.append(img)

# Fetch the labels for the api sourced data
ax_goce.set_ylabel("AOL, deg")
ax_goce.set_yticks(np.arange(0, 360, 90))
ax_goce.set_xticks(np.arange(start_date.day, stop_date.day, 1))

ax_goce.set_title("GOCE")

markers=["s","o"]

for i, endpoint in enumerate(["nrlmsise00", "jb2008"]):

    ax_api = ax_nrlmsise00 if i==0 else ax_jb2008

    ax_api.set_title(endpoint.upper())

    url = args.hostname + "/api/" + endpoint

    print("WARNING: {} requests to Amentum API, may exceed quota".format(len(df_goce)))

    # Apply the function call onto each row of the dataframe
    res = df_goce.apply(fetch_density_from_api, args=(url,), axis=1)

    df_goce[endpoint] = [
        row['total_mass_density']['value'] for row in res.values
    ]

    # Prepare 2D API density data for plotting
    densities_api = stats.binned_statistic_2d(
        time_deltas,
        df_goce["argument_latitude"].values,
        df_goce[endpoint].values,
        statistic="median",
        bins=(tds, arg_lats),
    )

    img = ax_api.imshow(
        densities_api.statistic.T,
        extent=(start_date.day, stop_date.day, arg_lats.min(), arg_lats.max()),
        origin="lower",
        aspect="auto",
        cmap=plt.cm.jet,
        vmin=df_goce["density"].values.min(),
        vmax=df_goce["density"].values.max(),
    )

    images.append(img)

    # Set the labels for the api plots
    ax_api.set_ylabel("AOL, deg")
    ax_api.set_yticks(np.arange(0, 360, 90))
    ax_api.set_xticks(np.arange(start_date.day, stop_date.day, 1))

    # Now plot the profiles for a particular argument latitude
    ax_prof.plot(
        tds[:-1], 
        densities_api.statistic.T[midlat_index, :],
        label=endpoint.upper(),
        marker=markers[i]
    )

# Set x labels on bottom plot only
ax_api.set_xlabel("Day")

# Format colorbar axis
cb = fig_cont.colorbar(
    images[0], 
    ax=list((ax_goce, ax_nrlmsise00, ax_jb2008)), 
    format="%3.1e",
    fraction=0.1)

cb.set_label("Density " + r"$kgm^{-3}$")

fig_cont.savefig("Density_GOCE_vs_Models_{}.png".format(start_date.strftime("%Y%m%d")))

# draw the legend on the profile 
ax_prof.legend()
fig_prof.savefig("Density_vs_API_AOL_{}_{}.png".format(
    int(arg_lat_of_interest),
    start_date.strftime("%Y%m%d")))
