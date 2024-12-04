// This script processes a classified NAIP image into 30m resolution aggregated layers for each land cover class.
// The output layers are then exported as Earth Engine assets for further use in analysis.

// Load the NAIP classified image (1m resolution) containing 7 land cover classes of Maryland. Do the same for DE and VA. 
var naip = NAIP_MD;  // Replace NAIP_MD with the actual NAIP image variable

// Define target resolution for aggregation (30 meters)
var targetResolution = 30;

// Function to aggregate land cover classes into 30m resolution
function aggregateClass(naipImage, classValue) {
  // Create a binary mask for pixels of the specified class
  var classMask = naipImage.eq(classValue).selfMask();

  // Aggregate the mask to 30m resolution by counting pixels within each 30m grid
  var aggregated = classMask.reduceResolution({
    reducer: ee.Reducer.count(), // Use count to aggregate
    bestEffort: false,
    maxPixels: 900 // Maximum pixels for a 30m x 30m cell (900 1m pixels)
  }).reproject({
    crs: naip.projection(),
    scale: targetResolution
  }).rename('class_' + classValue);

  return aggregated;
}

// Initialize an empty image for stacking aggregated land cover classes
var aggregatedImage = ee.Image([]);

// Loop over the land cover class values (1 to 8) and aggregate each class
for (var i = 1; i <= 8; i++) {
  var aggregatedClass = aggregateClass(naip, i);
  aggregatedImage = aggregatedImage.addBands(aggregatedClass);
}

// Filter pixels that have a count equal to 900 (pure 30m pixels) and convert them to 1
// for each land cover class layer (classes 1 to 8)
var retainedClasses = [];
for (var i = 1; i <= 8; i++) {
  var retained = aggregatedImage.select('class_' + i)
    .updateMask(aggregatedImage.select('class_' + i).gte(900)) // Retain pixels with count >= 900
    .where(aggregatedImage.select('class_' + i).gte(900), 1) // Convert to 1
    .toUint8();  // Convert to unsigned 8-bit integer
  retainedClasses.push(retained); // Store the retained layer
  Map.addLayer(retained, {}, 'Class ' + i);  // Add layer to map for visualization
}

// Export retained layers (one for each land cover class) as Earth Engine assets
var landCoverClasses = ['Forest', 'Marsh', 'Built', 'Water', 'Farmland', 'BareSoil', 'OtherVeg'];

// Loop through each retained class and export to Earth Engine
for (var i = 0; i < retainedClasses.length; i++) {
  Export.image.toAsset({
    image: retainedClasses[i],
    description: 'MD_NAIP_Retained_' + landCoverClasses[i],  // Dynamic description based on class name
    assetId: 'MD_NAIP_Retained_' + landCoverClasses[i],  // Dynamic asset ID based on class name
    scale: targetResolution,  // Resolution set to 30 meters
    region: NAIP_MD.geometry(),  // Define region of interest (replace with your region geometry)
    maxPixels: 1e13  // Max number of pixels to export
  });
}

