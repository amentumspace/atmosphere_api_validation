# Background

The US Standard Atmosphere is a model that describes the variation of temperature, pressure, and density with altitude. It was first published in 1958 and last updated in 1976. 

The model separates the atmosphere into regions with linear, or constant, temperature variation with geopotential altitude. The geopotential altitude accounts for the variation of acceleration due to gravity with geometric height. The variation of pressure with geopotential altitude is calculated by solving the hydrostatic equation. The density is calculated assuming an ideal gas law. 

Herein we implement the US Standard Atmosphere 1976 up to a geometric altitude of 86 km in Python code. We benchmark temperature and density values predicted by NRLMSISE-00, as obtained with the Amentum API, to values predicted by the US Standard Atmosphere.

More information on the US Standard Atmosphere can be found in the original technical report [here](https://ntrs.nasa.gov/search.jsp?R=19770009539). 

# Run time environment

Install the required Python packages using pip like so

    pip install -r requirements.txt 

# Run the analysis 

    python analysis.py

    usage: analysis.py [-h] [--host HOST] [--api_key API_KEY]

    optional arguments:
    -h, --help         show this help message and exit
    --host HOST        specify an alternative hostname for testing (e.g. on-premises server)
    --api_key API_KEY  valid API key obtained from https://developer.amentum.io

# Results

![](./air_temperature_profile.png)

Figure 1: Temperature profile for geometric altitudes less than 86 km as prediced by the US Standard Atmosphere (1976) and the NRLMSISE-00 model.

 ![](./air_density_profile.png)

 Figure 2: Same as Figure 1 for mass density.

Copyright 2021 [Amentum Aerospace](https://amentum.space), Australia.

