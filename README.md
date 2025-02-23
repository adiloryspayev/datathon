# City Services Analysis and Prediction Model

## Project Overview
This project analyzes city service data and develops machine learning models to predict service types and response times. It combines parcel data, parking information, and city service line data to create a comprehensive understanding of urban service patterns and needs.

## Key Features
- Multi-task neural network for predicting service types and response times
- Geospatial analysis of service distributions
- Sewer backup hotspot identification
- Interactive visualization of predictions
- Service time prediction model

## Data Sources
The project utilizes three main datasets:
- `parcel.csv`: Property parcel information including condition ratings
- `parking.csv`: Parking-related data
- `cityline.csv`: City service request records

## Key Findings

### Sewer Backup Analysis
Our analysis revealed several hotspots for sewer backup issues:
- Generated heatmap visualization showing concentration of incidents
- Identified seasonal patterns in service requests
- Top 10 locations account for approximately 30% of all backup reports

### Service Response Time Patterns
The multi-task model revealed:
- Average response times vary significantly by service type
- Geographic location influences service delivery speed
- Seasonal variations in service request volumes and response times

### Model Performance
Our multi-task neural network achieved:
- Service Type Prediction Accuracy: ~85%
- Response Time Category Prediction Accuracy: ~78%
- Strong performance in identifying high-priority service areas

## Technical Architecture

### Data Processing Pipeline
```python
def merge_and_preprocess_data(df1, df3):
    # Clean coordinates
    df1 = df1.dropna(subset=['Lat', 'Lng'])
    df3 = df3.dropna(subset=['Lat', 'Lng'])
    
    # Build KDTree for spatial joining
    tree = cKDTree(df1[['Lat', 'Lng']].values)
    distances, indices = tree.query(df3[['Lat', 'Lng']].values, k=1)
    
    # Merge datasets
    df_combined = df3.join(df1.iloc[indices], rsuffix='_df1')
```

### Model Architecture
```python
class MultiTaskServiceTimeModel(nn.Module):
    def __init__(self, input_dim, num_service_classes, num_time_classes=3):
        super().__init__()
        self.shared_layer1 = nn.Linear(input_dim, 64)
        self.shared_layer2 = nn.Linear(64, 32)
        self.service_layer = nn.Linear(32, num_service_classes)
        self.time_layer = nn.Linear(32, num_time_classes)
```

## Visualizations

### Sewer Backup Heatmap
The heatmap visualization reveals concentrated areas of sewer backup issues:
```python
# Create Folium Map
m = folium.Map(location=[43.0481, -76.1474], zoom_start=12)
HeatMap(locations, radius=15, blur=10, min_opacity=0.5).add_to(m)
```

### Service Prediction Distribution
```python
def visualize_predictions(X_test_df, y_pred_test):
    plt.scatter(X_test_df[y_pred_test == 1]['LONG'],
                X_test_df[y_pred_test == 1]['LAT'],
                color='red', label='High Priority')
    plt.scatter(X_test_df[y_pred_test == 0]['LONG'],
                X_test_df[y_pred_test == 0]['LAT'],
                color='green', label='Normal Priority')
```

## Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/city-services-prediction.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the main analysis:
```bash
python main.py
```

4. For predictions:
```python
from predictor import predict_service_and_time
address = "123 Example St, Syracuse, NY"
predict_service_and_time(address, model, scaler, imputer, le_service)
```

## Requirements
- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Folium
- Matplotlib
- Geopy

## Future Improvements
1. Integration with real-time service request data
2. Enhanced visualization dashboard
3. Mobile application for field workers
4. Automated alert system for high-priority areas
5. Integration with city's infrastructure maintenance schedule

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
