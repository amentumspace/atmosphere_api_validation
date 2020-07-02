import argparse
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from scipy import stats
from requests_futures.sessions import FuturesSession

"""

Comparing atmospheric density measured by GOCE to that predicted by 
NRLMSISE00 and JB2008 models accessed via the Amentum Atmosphere API. 

"""

# handle command line arguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host",
    dest="host",
    action="store",
    help="specify an alternative host for testing (e.g. on-premises)",
    default="https://atmosphere.amentum.space",
)
parser.add_argument(
    "--goce_file",
    dest="goce_file",
    action="store",
    help="specify path to GOCE density and wind data time series (goce_denswind_ac082_v2_0_YYYY-MM.txt)",
    required=True,
)

args = parser.parse_args()

# check for cached results if input file has already been processed

# base of filename to be used to save image and cached dataframe
dirname = os.path.dirname(args.goce_file)
basename = os.path.basename(args.goce_file)

pickled_filename = dirname+basename.replace(".txt",".pkl")
if os.path.isfile(pickled_filename):
    df_goce = pd.read_pickle(pickled_filename)
else: 

    # otherwise read it in again
    df_goce = pd.read_csv(args.goce_file, sep="\s+", comment="#", header=None)

    if len(df_goce) > 1e4:
        print(
            "WARNING: requests to Amentum API will exceed quota. limiting to 10K datapoints"
        )
        df_goce = df_goce.iloc[:10000]

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
    df_goce["datetime"] = df_goce.date + " " + df_goce.time

    # Convert date and time to datetime object to enable filtering of data based thereuponst
    df_goce["datetime"] = pd.to_datetime(df_goce.datetime)

    # Calculate NRLMSISE-00 model densities using the API

    def get_futures_request(row, session, url):
        """
        Creates futures requests to sample the atmospheric density 
        using the NRLMSISE-00 or JB2008 model
        
        Args: 
            row of pandas dataframe containing conditions at time of measurement
            url of the end point to hit
        Returns:
            session
        
        """
        payload = {
            "altitude": row["altitude"] / 1000.0,  # convert to kms
            "geodetic_latitude": row["latitude"],  # -90 to 90
            "geodetic_longitude": row["longitude"],  # 0 to 360
            "year": row["datetime"].year,
            "month": row["datetime"].month,
            "day": row["datetime"].day,
            "utc": row["datetime"].hour + row["datetime"].minute / 60,  # decimal UTC hour
        }

        return session.get(url, params=payload)

    def process_futures_request(future):
        """
        Process the futures request checking for errors and return value 
        from response.
        """
        try:
            response = future.result()
        except requests.exceptions.RequestException as e:
            assert False, e.args
        # make sure our response is ok
        assert response.ok, response.text
        # return the value received from the server
        return response.json()

    for i, endpoint in enumerate(["nrlmsise00", "jb2008"]):

        url = args.host + "/api/" + endpoint

        print(f"[*] Fetching API data from {url}")

        session = FuturesSession()

        requests = []
        # Apply the function call onto each row of the dataframe
        print(f"[*] Creating futures requests")
        requests = df_goce.apply(get_futures_request, args=(session, url,), axis=1)

        print(f"[*] Processing futures requests")
        responses = [process_futures_request(request) for request in requests]

        print(f"[*] Parsing responses")
        df_goce[endpoint] = [res["total_mass_density"]["value"] for res in responses]

    # now cache to data directory
    df_goce.to_pickle(pickled_filename)

# Visualise 

# limits for binning of timestamp and arg of lat
# we calculate dsitributions of mean density for discrete values of seconds from
# the start date, and discrete values of argument of latitude.
# the resolution can be tuned to reduce number of API calls for the study
time_delta_low = 0
time_delta_high = (df_goce.datetime.max() - df_goce.datetime.min()).total_seconds()

# go for hourly or daily bins.
mins_per_bin = 120
seconds_per_bin = 60 * mins_per_bin

tds = np.arange(time_delta_low, time_delta_high, seconds_per_bin)

# From wikipedia: the argument of latitude is an angular parameter that defines the 
# position of a body moving along a Kepler orbit. It is the angle 
# between the ascending node and the body. It is the sum of the 
# more commonly used true anomaly and argument of periapsis
arg_lat_delta = 18  # argument of latitude resolution in degrees
arg_lats = np.arange(0, 360 + arg_lat_delta, arg_lat_delta)

# Convert datetimes to delta time in seconds since first measurements
# This will be used for the binning and plotting
time_deltas = df_goce.datetime.values - df_goce.datetime.values.min()

# Convert time_deltas to seconds, will also convert to float type
time_deltas = [t / np.timedelta64(1, "s") for t in time_deltas]

# Calculate the GOCE densities as mean values lying within 2d grid of bins
densities = stats.binned_statistic_2d(
    time_deltas,
    df_goce.argument_latitude.values,
    df_goce.density.values,
    statistic="median",
    bins=(tds, arg_lats),
)

# get start/stop timestamps
start_date = df_goce.datetime.min()
stop_date = df_goce.datetime.max()

# plot at particular argument of latitude

fig_prof = plt.figure()
ax_prof = fig_prof.add_subplot(111)
# set year and month (as word) as x axis label
ax_prof.set_xlabel(start_date.strftime("%B %Y"))
ax_prof.set_ylabel("Density " + r"$kgm^{-3}$")

midlat_index = np.searchsorted(arg_lats, 180)
arg_lat_of_interest = arg_lats[midlat_index]

#  plot GOCE data
ax_prof.plot(tds[:-1], densities.statistic.T[midlat_index, :], label="GOCE", marker="D")

# labels = [item.get_text() for item in ax_prof.get_xticklabels()]

fig_prof.suptitle(
    "Median Density for AOL {}-{} deg".format(
        arg_lat_of_interest, arg_lats[midlat_index + 1]
    ),
    fontsize=12,
)

# initialise contour figure 

fig_cont, (ax_goce, ax_nrlmsise00, ax_jb2008) = plt.subplots(
    nrows=3, sharex=True, figsize=(8, 12)
)
fig_cont.suptitle(start_date.strftime("%B %Y"))

images = []

# plot 2D median density as measured by GOCE

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

ax_goce.set_ylabel("AOL, deg")
ax_goce.set_yticks(np.arange(0, 360, 90))
# ax_goce.set_xticks(np.arange(start_date.day, stop_date.day, 1))

ax_goce.set_title("GOCE")

markers = ["s", "o"]

# Plot model data at same time stamp / argument of latitude coords

for i, endpoint in enumerate(["nrlmsise00", "jb2008"]):

    ax_api = ax_nrlmsise00 if i == 0 else ax_jb2008

    ax_api.set_title(endpoint.upper())

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

    # Now plot the model profiles for a particular argument latitude
    ax_prof.plot(
        tds[:-1],
        densities_api.statistic.T[midlat_index, :],
        label=endpoint.upper(),
        marker=markers[i],
    )

# Set x labels on bottom plot only
ax_api.set_xlabel("Day")

# Format colorbar axis
cb = fig_cont.colorbar(
    images[0],
    ax=list((ax_goce, ax_nrlmsise00, ax_jb2008)),
    format="%3.1e",
    fraction=0.1,
)

cb.set_label("Density " + r"$kgm^{-3}$")

fig_cont.savefig(dirname+basename.replace(".txt","_cont.png"))

# draw the legend on the profile
ax_prof.legend()

fig_prof.savefig(dirname+basename.replace(".txt","_prof.png"))

