# Atmosphere API Validation studies

This repository contains validation studies for the Atmosphere API available at https://amentum.space 

Each directory contains: 

- a README.md file containing a description of the study, summary of findings, and instructions to run the Python code and re-generate the results
- images or profiles comparing values of atmospheric density or temperature obtained using the Atmosphere API alongside experimental results or predictions of other models.

Feel free to add a validation study by creating a new branch and submitting a pull request. 

# Running the analyses

See the analysis.py script in each directory to see how the Amentum Atmosphere API was used to retrieve density and temperature values from the NRLMSISE-00 model, as well as the Python code to fetch and process validation data from experimental measurements, or benchmarking data from another model.

Install the required Python packages using pip like so

    pip install -r requirements.txt 

Then run the script using the following command

    python analysis.py 

That will produce results as PNG files in the same directory. 

Copyright 2020 [Amentum Aerospace](https://amentum.space), Australia.
