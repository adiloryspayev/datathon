
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
df1 = pd.read_csv("parcel.csv")
df2 = pd.read_csv("parking.csv")
df3 = pd.read_csv("cityline.csv")
# Step 1: Merge and Preprocess Data
def merge_and_preprocess_data(df1, df3):
    print("Starting preprocessing...")
    # Clean df1 coordinates
    df1 = df1.dropna(subset=['Lat', 'Lng'])
    df1 = df1[~df1[['Lat', 'Lng']].isin([float('inf'), float('-inf')]).any(axis=1)]
    print(f"df1 shape after cleaning: {df1.shape}")

    # Clean df3 coordinates
    df3 = df3.dropna(subset=['Lat', 'Lng'])
    df3 = df3[~df3[['Lat', 'Lng']].isin([float('inf'), float('-inf')]).any(axis=1)]
    print(f"df3 shape after cleaning: {df3.shape}")

    if df1.empty or df3.empty:
        raise ValueError("No valid data left after cleaning coordinates.")

    # Build a KDTree for df1 coordinates
    lat_lng_values = df1[['Lat', 'Lng']].values
    if not isinstance(lat_lng_values, np.ndarray):
        raise TypeError("df1[['Lat', 'Lng']].values is not a numpy array")
    tree = cKDTree(lat_lng_values)
    query_values = df3[['Lat', 'Lng']].values
    if not isinstance(query_values, np.ndarray):
        raise TypeError("df3[['Lat', 'Lng']].values is not a numpy array")
    distances, indices = tree.query(query_values, k=1)
    df3['Nearest_df1_index'] = indices

    # Merge nearest neighbor data
    df_combined = df3.join(df1.iloc[df3['Nearest_df1_index']], rsuffix='_df1')
    print(f"df_combined shape: {df_combined.shape}")

    # Feature and target setup
    features = ['Lat', 'Lng', 'Rating', 'Minutes_to_Acknowledge', 'Minutes_to_Close', 'Sla_in_hours',
                'SHAPE_Leng', 'SHAPE_Area']

    # Check if required columns exist
    missing_cols = [col for col in features if col not in df_combined.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in merged data: {missing_cols}")

    le_service = LabelEncoder()
    df_combined['ServiceType'] = le_service.fit_transform(df_combined['Request_type'])

    # Binning time
    df_combined['TimeToAttention'] = pd.qcut(
        df_combined['Minutes_to_Acknowledge'].fillna(df_combined['Sla_in_hours'] * 60),
        q=3, labels=[0, 1, 2], duplicates='drop'
    ).cat.codes

    df = df_combined.dropna(subset=features + ['ServiceType', 'TimeToAttention'])
    print(f"Final df shape: {df.shape}")

    if df.empty:
        raise ValueError("No valid data left after dropping NaN values.")

    X = df[features]
    y_service = df['ServiceType']
    y_time = df['TimeToAttention']

    X_train, X_test, y_service_train, y_service_test, y_time_train, y_time_test = train_test_split(
        X, y_service, y_time, test_size=0.2, random_state=42
    )

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    return (torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_service_train.values, dtype=torch.long),
            torch.tensor(y_service_test.values, dtype=torch.long),
            torch.tensor(y_time_train.values, dtype=torch.long),
            torch.tensor(y_time_test.values, dtype=torch.long),
            X_test, scaler, imputer, le_service)

# Step 2: Define Multi-Task Model
class MultiTaskServiceTimeModel(nn.Module):
    def __init__(self, input_dim, num_service_classes, num_time_classes=3):
        super().__init__()
        self.shared_layer1 = nn.Linear(input_dim, 64)
        self.shared_layer2 = nn.Linear(64, 32)
        self.service_layer = nn.Linear(32, num_service_classes)
        self.time_layer = nn.Linear(32, num_time_classes)

    def forward(self, x):
        x = torch.relu(self.shared_layer1(x))
        x = torch.relu(self.shared_layer2(x))
        return self.service_layer(x), self.time_layer(x)

# Step 3: Train Model
def train_model(model, X_train, y_service_train, y_time_train, criterion, optimizer, epochs=2000):
    losses = []
    for epoch in range(epochs):
        service_out, time_out = model(X_train)
        loss_service = criterion(service_out, y_service_train)
        loss_time = criterion(time_out, y_time_train)
        total_loss = loss_service + loss_time

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}')

    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Over Time')
    plt.show()

# Step 4: Evaluate Model
def evaluate_model(model, X_test, y_service_test, y_time_test, le_service):
    with torch.no_grad():
        service_pred, time_pred = model(X_test)
        service_pred = torch.argmax(service_pred, dim=1)
        time_pred = torch.argmax(time_pred, dim=1)

    unique_classes = torch.unique(y_service_test).numpy()
    service_names = [str(le_service.classes_[int(i)]) for i in unique_classes]

    print("Service Accuracy:", accuracy_score(y_service_test, service_pred))
    print("Time Accuracy:", accuracy_score(y_time_test, time_pred))
    print("\nService Report:\n", classification_report(y_service_test, service_pred, target_names=service_names, zero_division=1))
    print("Time Report:\n", classification_report(y_time_test, time_pred, target_names=['soon', 'medium', 'later'], zero_division=1))

    return service_pred, time_pred

# Main function
def main(df1, df3):
    try:

        # Rename columns for consistency
        df1 = df1.rename(columns={'LAT': 'Lat', 'LONG': 'Lng'})

        # Preprocess data
        (X_train, X_test, y_service_train, y_service_test,
         y_time_train, y_time_test, X_test_raw, scaler, imputer, le_service) = merge_and_preprocess_data(df1, df3)

        # Initialize model
        input_dim = X_train.shape[1]
        num_service_classes = len(le_service.classes_)
        model = MultiTaskServiceTimeModel(input_dim, num_service_classes)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        train_model(model, X_train, y_service_train, y_time_train, criterion, optimizer, epochs=2000)

        # Evaluate model
        evaluate_model(model, X_test, y_service_test, y_time_test, le_service)

        # Save the model and preprocessing tools with metadata
        try:
            metadata = {
                'model_state': model.state_dict(),
                'input_dim': input_dim,
                'num_service_classes': num_service_classes
            }
            torch.save(metadata, 'service_time_model_v1.pt')
            import pickle
            with open('scaler_v1.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open('imputer_v1.pkl', 'wb') as f:
                pickle.dump(imputer, f)
            with open('label_encoder_v1.pkl', 'wb') as f:
                pickle.dump(le_service, f)
            print("Model and tools saved: service_time_model_v1.pt, scaler_v1.pkl, imputer_v1.pkl, label_encoder_v1.pkl")
        except Exception as e:
            print(f"Error saving model and tools: {e}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file(s) - {e}")
    except ValueError as e:
        print(f"Error in data processing: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    input_dim = X_train.shape[1]
    num_service_classes = len(le_service.classes_)
    model = MultiTaskServiceTimeModel(input_dim, num_service_classes)

        # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
    train_model(model, X_train, y_service_train, y_time_train, criterion, optimizer, epochs=2000)

        # Evaluate model
    evaluate_model(model, X_test, y_service_test, y_time_test, le_service)

        # Save the model and preprocessing tools with metadata
        # Existing save code...

    return model  # Return the trained model


# Main execution
if __name__ == "__main__":
    model = main(df1, df3)  # Store the returned model
