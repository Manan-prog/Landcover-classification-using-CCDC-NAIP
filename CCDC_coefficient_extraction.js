// ============================
// CCDC Coefficient Calculation
// ============================

// Load the required utility API for CCDC processing
var utils = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/api');

// Load and mosaic the CCDC results image collection
var ccdResultsCollection = ee.ImageCollection('projects/CCDC/v3');
var ccdResults = ccdResultsCollection.mosaic();

// Define the time range for filtering
var startYear = 2016;
var endYear = 2018;

// Select the time of break and create a mask for no breaks between the time range
var change = ccdResults.select('tBreak');
var noBreakMask = change.lt(startYear).or(change.gt(endYear));

// Apply the no-break mask to the CCDC results
var noBreakImage = ccdResults.updateMask(noBreakMask);

// Select coefficient bands for NIR and RED and extract slope values
var nirCoefs = ccdResults.select('NIR_coefs');
var redCoefs = ccdResults.select('RED_coefs');
var nirSlope = nirCoefs.arraySlice(1, 1, 2).arrayProject([0]).rename('NIR_Slope');
var redSlope = redCoefs.arraySlice(1, 1, 2).arrayProject([0]).rename('RED_Slope');

// Create a mask for slope values within the range [-0.005, 0.005]
var slopeMask = nirSlope.gte(-0.005).and(nirSlope.lte(0.005))
                         .and(redSlope.gte(-0.005)).and(redSlope.lte(0.005));

// Combine masks for the final filtered image
var finalMask = noBreakMask.and(slopeMask);
var CCDC_MaskfilteredImage = noBreakImage.updateMask(finalMask);

// Extract the first element from the 'tBreak' band array for scalar masking
var finalMask_band = finalMask.select('tBreak');
var FinalMask_scalarBand = finalMask_band.arrayGet(0).rename('scalar_bandMask');

// ============================
// Land Cover Sampling
// ============================

// Function to sample and label land cover points
function sampleLandCover(region, classBand, landCoverLabel) {
  var samples = region.updateMask(FinalMask_scalarBand).stratifiedSample({
    numPoints: 2000,
    classBand: classBand,
    region: region.geometry(),
    scale: 30,
    geometries: true,
    seed: 13
  });

  return samples.map(function(feature) {
    return feature.set('Land Cover', landCoverLabel);
  });
}

// Sample and label points for each land cover type
var Forest_Samples = sampleLandCover(Forest, 'class_1', 'Forest');
var Marsh_Samples = sampleLandCover(Marsh, 'class_2', 'Marsh');
var Built_Samples = sampleLandCover(Built, 'class_4', 'Built');
var Water_Samples = sampleLandCover(Water, 'class_5', 'Water');
var Farmland_Samples = sampleLandCover(Farmland, 'class_6', 'Farmland');
var BareSoil_Samples = sampleLandCover(BareSoil, 'class_7', 'BareSoil');
var OtherVegetation_Samples = sampleLandCover(OtherVegetation, 'class_8', 'OtherVegetation');

// ============================
// Synthetic Data Sampling
// ============================

// Define spectral band names and temporal segments
var BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'];
var SEGS = [
  "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
  "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20",
  "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30"
];

// Obtain CCDC results in 'regular' ee.Image format
var ccdImage = utils.CCDC.buildCcdImage(ccdResults, SEGS.length, BANDS)


// Define functions for calculating indices

// Calculate NDVI
function calcNDVI(image) {
  var ndvi = ee.Image(image).normalizedDifference(['NIR', 'RED']).rename('NDVI');
  return ndvi;
}

// Calculate NBR
function calcNBR(image) {
  var nbr = ee.Image(image).normalizedDifference(['NIR', 'SWIR2']).rename('NBR');
  return nbr;
}

// Calculate EVI
function calcEVI(image) {
  var evi = ee.Image(image).expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('NIR'),
      'RED': image.select('RED'),
      'BLUE': image.select('BLUE')
    }).rename('EVI');
  return evi;
}

// Calculate EVI2
function calcEVI2(image) {
  var evi2 = ee.Image(image).expression(
    '2.5 * ((NIR - RED) / (NIR + 2.4 * RED + 1))', {
      'NIR': image.select('NIR'),
      'RED': image.select('RED')
    }).rename('EVI2');
  return evi2;
}

// Tassel Cap Transformation
function tcTrans(image) {
  var brightness = image.expression(
    '0.2043 * BLUE + 0.4158 * GREEN + 0.5524 * RED + 0.5741 * NIR + 0.3124 * SWIR1 + 0.2303 * SWIR2', {
      'BLUE': image.select('BLUE'),
      'GREEN': image.select('GREEN'),
      'RED': image.select('RED'),
      'NIR': image.select('NIR'),
      'SWIR1': image.select('SWIR1'),
      'SWIR2': image.select('SWIR2')
    }).rename('BRIGHTNESS');

  var greenness = image.expression(
    '-0.1603 * BLUE - 0.2819 * GREEN - 0.4934 * RED + 0.7940 * NIR - 0.0002 * SWIR1 - 0.1446 * SWIR2', {
      'BLUE': image.select('BLUE'),
      'GREEN': image.select('GREEN'),
      'RED': image.select('RED'),
      'NIR': image.select('NIR'),
      'SWIR1': image.select('SWIR1'),
      'SWIR2': image.select('SWIR2')
    }).rename('GREENNESS');

  var wetness = image.expression(
    '0.0315 * BLUE + 0.2021 * GREEN + 0.3102 * RED + 0.1594 * NIR - 0.6806 * SWIR1 - 0.6109 * SWIR2', {
      'BLUE': image.select('BLUE'),
      'GREEN': image.select('GREEN'),
      'RED': image.select('RED'),
      'NIR': image.select('NIR'),
      'SWIR1': image.select('SWIR1'),
      'SWIR2': image.select('SWIR2')
    }).rename('WETNESS');

  return ee.Image([brightness, greenness, wetness]);
}

// Calculate indices and add as bands
function addIndices(image) {
  var ndvi = calcNDVI(image);
  var nbr = calcNBR(image);
  var evi = calcEVI(image);
  var evi2 = calcEVI2(image);
  var tc = tcTrans(image);
  
  return image.addBands([ndvi, nbr, evi, evi2, tc]);
}



// Function to rename bands
var renameBands = function(image, suffix) {
  var bandNames = image.bandNames();
  var newBandNames = bandNames.map(function(band) {
    return ee.String(band).cat('_').cat(suffix);
  });
  return image.rename(newBandNames);
};

// Generate synthetic images for specific months and select bands
var getSyntheticBands = function(date, monthName) {
  var dateParams = {inputFormat: 3, inputDate: date, outputFormat: 1};
  var formattedDate = utils.Dates.convertDate(dateParams);
  var syntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS);
  var syntheticBands = addIndices(syntheticBands)
  return renameBands(syntheticBands, monthName);
};

// Process synthetic images for February, May, August, and November
var Jan_SyntheticBands = getSyntheticBands('2017-01-01', 'Jan');
var Apr_SyntheticBands = getSyntheticBands('2017-04-01', 'Apr');
var Jul_SyntheticBands = getSyntheticBands('2017-07-01', 'Jul');
var Oct_SyntheticBands = getSyntheticBands('2017-10-01', 'Oct');

// Stack the selected images together
var stackedImage = Jan_SyntheticBands
  .addBands(Apr_SyntheticBands)
  .addBands(Jul_SyntheticBands)
  .addBands(Oct_SyntheticBands).clip(NAIP_MD.geometry());
  
  
Map.addLayer(stackedImage,{},'Stacked Image')

// Merge all feature collections into one FeatureCollection with a 'class' property
var samples = Forest_Samples.merge(Marsh_Samples).merge(Built_Samples)
.merge(Water_Samples).merge(Farmland_Samples).merge(BareSoil_Samples)
.merge(OtherVegetation_Samples)

print(samples,'Samples')

// Sample the raster using the feature collection. Specify the scale (e.g., 30 meters)
var sampledData = stackedImage.sampleRegions({
  collection: samples, 
  properties: ['Land Cover'], // Keep the 'class' property
  scale: 30, // Adjust to the resolution of your raster
  tileScale: 16 // Use larger tileScale if you are working with large rasters to optimize memory use
});

print(sampledData, 'Final Sampled data')

// Export the data as a CSV to Google Drive
Export.table.toDrive({
  collection: sampledData,
  description: 'LandCover_TrainingData',
  fileFormat: 'CSV'
});
