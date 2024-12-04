# Landcover-classification-using-CCDC-NAIP
The code here is an integration of scripts from Google Earth Engine (GEE) and Python. GEE was used to sample and extract training data and Python was used to train a random forest classifier to perform land cover classification. 

The Continuous Change Detection and Classification (CCDC) [Zhe Zhu and Curtis Woodcock, 2014] algorithm was used to derive synthetic Landsat imagery for land cover classification. 
**CCDC dataset:** https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_GLOBAL_CCDC_V1

CCDC is a powerful method for detecting and analyzing land surface changes over time by leveraging the full range of available Landsat data. It builds a time series model for each 30m pixel using cloud-free observations, effectively capturing baseline conditions. Changes are detected by comparing the predicted values from the model with actual satellite observations, identifying deviations beyond a set threshold within a moving time window. When such deviations occur, a change is recorded, which may indicate shifts in land cover, vegetation health, or other surface characteristics. The CCDC algorithm then recalibrates the model to reflect the new conditions, allowing it to track both abrupt and gradual land cover transitions, such as deforestation or vegetation regrowth. The key parameters of the model include intercept (baseline value), slope (rate of change), magnitude (intensity of change), and three harmonic components that model annual, semi-annual, and finer periodic changes. Root mean square error (RMSE) is used to evaluate the model's fit accuracy. 

The seven land cover classes from the NAIP dataset were extracted into individual raster layers. Each layer, initially at a 1m spatial resolution, was aggregated to 30m resolution using the reducer function in Google Earth Engine
**NAIP dataset: ** https://data.niaid.nih.gov/resources?id=zenodo_6685694

**Land Cover Classification and Training Data Selection**

**1. Data Preparation:**
-  Seven land cover classes from NAIP data were extracted into individual raster layers.
- Layers (1m resolution) were aggregated to 30m using Google Earth Engine's count function.
- Pixels with counts below 900 (non-pure pixels) were masked out.

**2. Stable Land Cover Selection:**
- Focused on pixels with stable land cover (no abrupt changes) for robust classification.
- Stability was defined as no land cover change within one year before or after the year of analysis.
- NAIP snapshots from summer limited the analysis to stable pixels over a three-year period.

**3. Regional Stability Windows:**
- Delaware & Maryland: Stable pixels from 2016-2018 using NAIP 2017 data.
- Virginia: Stable pixels from 2015-2017 using NAIP 2016 data.

**4. Slope Filtering:**
- Pixels with slope values between -0.005 and 0.005 (Red & NIR bands) were selected for stability.
- Red and NIR bands were used due to their sensitivity to vegetation-related classes.

**5. Final Mask Application:**
- Applied stability mask to aggregated land cover layers from the NAIP-Landsat fusion product.

**6. Stratified Sampling:**
- Selected 2000 stable points per land cover class across the Delmarva region using stratified random sampling.


### Preprocessing CCDC Data
1. **Seasonal Data Selection**:  
   - Selected Landsat images from the first dates of January, April, July, and October to represent winter, spring, summer, and fall.  
   - Incorporated both spectral and temporal dimensions to capture seasonal variability.

2. **Bands and Indices Computed**:  
   - Included basic Landsat bands: blue, green, red, NIR, SWIR1, SWIR2, and thermal.  
   - Calculated spectral indices to highlight landscape characteristics:  
     - **NDVI** and **EVI**: Vegetation health (sensitive to saline impacts).  
     - **NBR**: Surface moisture and stressed soil detection.  
     - **EVI2**: Alternative vegetation measure for consistency.

3. **Tasseled Cap Transformation**:  
   - Generated **Brightness**, **Greenness**, and **Wetness** bands.  
   - **Wetness** emphasized soil moisture levels, aiding in salinization impact detection.

4. **Stacked Image Creation**:  
   - Combined bands, indices, and tasseled cap coefficients into a single stacked image.  
   - Captured a comprehensive temporal snapshot relevant to seasonal and salinization analysis.

5. **Data Extraction for Classification**:  
   - Used the stacked image to extract data over selected sample points for each land cover class.

6. **Implementation in Google Earth Engine**:  
   - Conducted all preprocessing and analysis in Google Earth Engine for computational efficiency and spatial consistency.  

7. **Outcome**:  
   - Enabled detection of subtle seasonal and salinization-driven changes in landscape conditions.

Lastly, a random forest classifier was trained using the reference data.
Now the model is ready to be applied for any year from Landsat time series for land cover classification. 
