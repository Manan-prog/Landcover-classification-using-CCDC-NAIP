import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import joblib
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the training dataset from CSV file
data_DE = pd.read_csv('2017_DE_TrainingData.csv')
data_MD = pd.read_csv('2017_MD_TrainingData.csv')
data_VA = pd.read_csv('2016_VA_TrainingData.csv')

appended_df = pd.concat([data_DE, data_MD, data_VA], ignore_index=True)

# Separate target variable (y) and predictor variables (X)
X = appended_df.drop(columns=['Land Cover','system:index','.geo'])
y = appended_df['Land Cover']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],          # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],         # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],         # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]            # Minimum number of samples required at each leaf node
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=3,  # Number of cross-validation folds
                           verbose=2, 
                           n_jobs=-1)  # Use all available cores

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validated Accuracy: {best_score * 100:.2f}%")

# Use the best parameters to train the Random Forest Classifier
best_rf_classifier = RandomForestClassifier(**best_params)
best_rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# Print classification report for detailed evaluation
print(classification_report(y_test, y_pred))

# Renaming the best model
rf_classifier = best_rf_classifier

# Save the model to a file
joblib_file = "random_forest_classifier.joblib"
joblib.dump(rf_classifier, joblib_file)
print(f"Model saved as {joblib_file}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create the ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf_classifier.classes_)
plt.figure(figsize=(10, 15))
# Plot the confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
# Save the figure with 300 DPI resolution
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.show()

# ================================
# Model application on unseen data
# ================================

# Path to your trained model
model_path = 'random_forest_classifier.joblib'

# Load the trained Random Forest classifier
rf_classifier = load(model_path)

# Define the correct order of bands for the model
correct_order = [
    'BLUE_Apr', 'BLUE_Jan', 'BLUE_Jul', 'BLUE_Oct', 'BRIGHTNESS_Apr',
    'BRIGHTNESS_Jan', 'BRIGHTNESS_Jul', 'BRIGHTNESS_Oct', 'EVI2_Apr',
    'EVI2_Jan', 'EVI2_Jul', 'EVI2_Oct', 'EVI_Apr', 'EVI_Jan', 'EVI_Jul',
    'EVI_Oct', 'GREENNESS_Apr', 'GREENNESS_Jan', 'GREENNESS_Jul',
    'GREENNESS_Oct', 'GREEN_Apr', 'GREEN_Jan', 'GREEN_Jul', 'GREEN_Oct',
    'NBR_Apr', 'NBR_Jan', 'NBR_Jul', 'NBR_Oct', 'NDVI_Apr', 'NDVI_Jan',
    'NDVI_Jul', 'NDVI_Oct', 'NIR_Apr', 'NIR_Jan', 'NIR_Jul', 'NIR_Oct',
    'RED_Apr', 'RED_Jan', 'RED_Jul', 'RED_Oct', 'SWIR1_Apr', 'SWIR1_Jan',
    'SWIR1_Jul', 'SWIR1_Oct', 'SWIR2_Apr', 'SWIR2_Jan', 'SWIR2_Jul',
    'SWIR2_Oct', 'TEMP_Apr', 'TEMP_Jan', 'TEMP_Jul', 'TEMP_Oct',
    'WETNESS_Apr', 'WETNESS_Jan', 'WETNESS_Jul', 'WETNESS_Oct'
]

# Path to the input raster you want to classify
input_raster_path = 'Path_to_raster_to_be_classified.tif'

# Open the raster using rasterio
with rasterio.open(input_raster_path) as src:
    # Check if band descriptions are available
    band_descriptions = [src.descriptions[i] if src.descriptions[i] else f'Band {i+1}' for i in range(src.count)]
    
    # Print the band descriptions to verify their order
    print("Band Descriptions:", band_descriptions)

    # Reorder the bands to match the correct order if descriptions are available
    if len(band_descriptions) > 0 and all(band in band_descriptions for band in correct_order):
        band_indices = [band_descriptions.index(band) + 1 for band in correct_order]  # 1-based index for bands
        img = np.stack([src.read(i) for i in band_indices])
    else:
        print("Band descriptions do not match, assuming correct order by default.")
        img = src.read()  # Assuming the bands are already in the correct order

# Reshape the image for classification
img_reshaped = np.moveaxis(img, 0, -1).reshape(-1, img.shape[0])  # Reshape to (n_samples, n_features)

# Create a mask for NaN values
nan_mask = np.any(np.isnan(img), axis=0)

# Apply the classifier to the reshaped data, ignoring NaN values
img_reshaped_no_nan = img_reshaped[~np.isnan(img_reshaped).any(axis=1)]
classified_no_nan = rf_classifier.predict(img_reshaped_no_nan)

# Map class names to integers
class_mapping = {
    'BareSoil': 0,
    'Built': 1,
    'Farmland': 2,
    'Forest': 3,
    'Marsh': 4,
    'OtherVegetation': 5,
    'Water': 6
}
classified_no_nan = np.vectorize(class_mapping.get)(classified_no_nan)

# Create a classified array with NaN for missing values
classified_reshaped = np.full((src.height, src.width), np.nan)  # Initialize with NaN
classified_reshaped[~nan_mask] = classified_no_nan  # Fill classified values where mask is False

# Save the classified output
output_raster_path = 'Give_desired_filename.tiff'
meta = src.meta
meta.update(driver='GTiff', count=1, dtype=rasterio.float32)

with rasterio.open(output_raster_path, 'w', **meta) as dst:
    dst.write(classified_reshaped, 1)

print(f"Classification complete. Output saved to: {output_raster_path}")


# ===================================
# Display classified raster
# ===================================

# Path to the classified raster
classified_raster_path = 'path_to_classified_raster.tiff'

# Define class names based on the provided mapping
class_names = {
    0: 'Bare Soil',
    1: 'Built',
    2: 'Farmland',
    3: 'Forest',
    4: 'Marsh',
    5: 'Other Vegetation',
    6: 'Water'
}

# Define a color map for the 7 classes (customize colors as needed)
cmap = plt.get_cmap('tab10', 7)  # Using a colormap with 7 distinct colors

# Open the classified raster using rasterio
with rasterio.open(classified_raster_path) as src:
    # Read the classified raster
    classified_data = src.read(1)  # Read the first (and only) band

    # Create a mask for NaN values
    classified_data_masked = np.ma.masked_where(np.isnan(classified_data), classified_data)

    # Display the classified raster
    plt.figure(figsize=(10, 6))
    plt.imshow(classified_data_masked, cmap=cmap, vmin=0, vmax=6)  # Assuming classes are labeled 0 to 6
    plt.title("Classified Raster with 7 Classes")
    
    # Create a legend using actual class names
    cbar = plt.colorbar(ticks=np.arange(7))  # Show colorbar with class labels
    cbar.ax.set_yticklabels([class_names[i] for i in range(7)])  # Set the labels to class names
    cbar.set_label('Classes')  # Label for the colorbar
    
    plt.axis('off')  # Hide axes for better visual presentation
    plt.show()
