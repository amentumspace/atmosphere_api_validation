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
Comparing with atmospheric density with that predicted by NRLMSISE00 and 
JB2008 models accessed via the Amentum Aerospace API

"""

# Obtain the hostname via command line argument (for on-premises deployment)
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
        "Date {} is outside mission data range {} to {}".format(
        args.start_date, mission_data_start, mission_data_stop)
    )

# TODO ensure date within range of mission

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

# Isolate 1 week 
start_date = args.start_date
stop_date = args.start_date + datetime.timedelta(weeks=1)

df_goce = df_goce[
    (df_goce["datetime"] >= start_date)
    & (df_goce["datetime"] < stop_date)
]

""" Reduce the dataset by only keeping every N-th sample
 reduces the number of API calls, but requires coarse binning. 
 TODO modify this for local testing and on-premises API installs"""
reduction_factor = 100
df_goce = df_goce.iloc[::reduction_factor, :]

# Create geomagnetic indices lookup dataframe for the month

current_month_str = start_date.strftime("%m")
next_month_str = (start_date + datetime.timedelta(days=31)).strftime("%m")
current_year_str = start_date.strftime("%y")

# get the indices for current and next month in case start date close to 
# month's end

df_Kp_list = []

for m in [current_month_str, next_month_str]:

    url = "ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/tab/"
    filename = "kp"+current_year_str+m+".tab"

    # file is cached, download if doesn't exist
    if not os.path.exists("./"+filename):
        urllib.request.urlretrieve(url+filename, "./"+filename)

    # Fetch geomagnetic indices from ftp server for the same year and month
    # One or more space is considered a separator
    df_Kp = pd.read_csv(
        "./"+filename, 
        sep="\s+", 
        comment="#", 
        header=None, 
        usecols=(0,10), # isolate date and Kp index,
        names=["date", "Ap"],
        skipfooter=4) # last lines are ignored

    # Convert date to datetime object
    # Will be used to look up geomag index on date of measurement
    df_Kp["date"] = pd.to_datetime(df_Kp["date"], format="%y%m%d")

    df_Kp_list.append(df_Kp)

df_Kp = pd.concat(df_Kp_list, axis=0, ignore_index=True)

# Look up the geomag index at each date of measurement
df_goce["Ap"] = [
    df_Kp["Ap"][df_Kp["date"].searchsorted(dt) - 1] for dt in df_goce.datetime.values
]

# Fetch radio flux lookup dataframe for the current and adjacent years
# to account for when we're close to month start or end

df_f107_list = []

# filename is in %Y format so we can jsut cast to str
for year in [start_date.year-1, start_date.year, start_date.year+1]:

    url = "ftp://ftp.swpc.noaa.gov/pub/warehouse/"+str(year)+"/"
    filename = str(year)+"_DSD.txt"

    # file is cached, download if doesn't exist
    if not os.path.exists("./"+filename):
        urllib.request.urlretrieve(url+filename, "./"+filename)

    # Fetch radio flux data for this year from local file
    df_f107 = pd.read_csv(
        "./"+filename, 
        sep="\s+", 
        comment="#", 
        header=None, 
        usecols=range(4),
        names=[
            "year",
            "month",
            "day",
            "radio_flux"
        ],
        skiprows=2)

    # Create new column of type datetime based on the ymd columns
    # Will be used to lookup radio flux on date of measurement
    df_f107["date"] = pd.to_datetime(df_f107[["year", "month", "day"]])

    # Drop unused columns
    df_f107 = df_f107.drop(
        columns=[
            "year",
            "month",
            "day",
        ]
    )

    df_f107_list.append(df_f107)

df_f107 = pd.concat(df_f107_list, axis=0, ignore_index=True)

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
# if there are enough datapoints either side, otherwise raise error
avg_flux_vals = []
for i, flux in enumerate(df_f107["radio_flux"].values):
    #
    if 40 < i < len(df_f107["radio_flux"].values) - 40:
        avg_flux_vals.append(np.mean(df_f107["radio_flux"].values[i - 40 : i + 40]))
    else:
        raise ValueError("Insufficient data either side of current date to calc average radio flux")

df_f107["radio_flux_avg"] = avg_flux_vals

# Look up the 81 day average radio flux at each date of measurement
df_goce["f107a"] = [
    df_f107["radio_flux_avg"][df_f107["date"].searchsorted(dt) - 1]
    for dt in df_goce.datetime.values
]

def fetch_density_from_api(row, url):
    """
    Make an API call to sample the atmospheric density using the 
    NRLMSISE-00 or JB2008 model
    
    Args: row of pandas dataframe containing conditions at time of measurement
    Returns:
        density in kg/m3
    
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
    # additional params required for the nrl endpoint
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
# point per bin with the sparse GOCE data to ensure daily API quote not exceeded 
# for users. Resolution can be improved for staging and on-premises deployments. 
time_delta_low = 0
time_delta_high = (stop_date - start_date).total_seconds()

# go for hourly or daily bins. 
seconds_per_hour = 60 * 60 
seconds_per_day = 60 * 60 * 24 
# bin to ensure final edge is considered
tds = np.arange(time_delta_low, time_delta_high+seconds_per_day, seconds_per_day)

arg_lat_delta = 10 # argument of latitude resolution in degrees
arg_lats = np.arange(0,360+arg_lat_delta,arg_lat_delta)

# Convert datetimes to delta since first measurements
# This will be used for the binning and plotting
# (avoids using datetime objects)
time_deltas = df_goce["datetime"].values - df_goce["datetime"].values.min()

# Convert time_deltas to seconds, will also convert to float type
time_deltas = [t / np.timedelta64(1, 's') for t in time_deltas]

# Calculate the GOCE densities as mean values lying within 2d grid of bins
densities = stats.binned_statistic_2d(
    time_deltas,
    df_goce["argument_latitude"].values,
    df_goce["density"].values,
    statistic="mean",
    bins=(tds, arg_lats),
)

# initialise the profile plot
fig_prof = plt.figure()
ax_prof = fig_prof.add_subplot(111)
ax_prof.set_xlabel(datetime.date(year, month, 1).strftime("%B %Y"))
ax_prof.set_ylabel("Density " + r"$kgm^{-3}$")

midlat_index = np.searchsorted(arg_lats, 180)

arg_lat_of_interest = arg_lats[midlat_index]

ax_prof.plot(tds[:-1], densities.statistic.T[midlat_index, :], label="GOCE")


labels = [item.get_text() for item in ax_prof.get_xticklabels()]

def format_func(value, tick_number):
    """
    Function to convert tick labels from seconds elapsed to 
    day of date.
    
    """
    return start_day + int(value / seconds_per_day)

ax_prof.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

fig_prof.suptitle(
    "Argument of latitude {0:.2f} deg".format(arg_lat_of_interest), fontsize=12
)

# Calculate NRLMSISE-00 model densities using the API

for endpoint in ["nrlmsise00", "jb2008"]:

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
    ax_prof.plot(
        tds[:-1], densities_api.statistic.T[midlat_index, :], label=endpoint.upper()
    )

ax_prof.legend()

fig_prof.savefig("Density_vs_API_AOL_{}.png".format(int(arg_lat_of_interest)))
