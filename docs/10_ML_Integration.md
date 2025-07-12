# ML Framework Integration

MDM provides seamless integration with popular machine learning frameworks including scikit-learn, PyTorch, TensorFlow, XGBoost, and others. This guide covers framework-specific integrations and best practices.

## Overview

MDM's ML integration features:
- **Framework Adapters**: Native data format conversion
- **Train/Test Management**: Automatic split handling
- **Feature Pipeline**: Integrated feature engineering
- **Submission Creation**: Competition-ready outputs
- **Cross-validation**: Time series and stratified splits
- **Model Tracking**: Optional MLflow integration

## Scikit-learn Integration

### Basic Usage

```python
from mdm import MDMClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

client = MDMClient()

# Load train/test split
X_train, X_test, y_train, y_test = client.ml.load_train_test_split(
    dataset="titanic",
    test_size=0.2,
    random_state=42,
    stratify=True  # Stratified split for classification
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### Advanced Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

# Create ML pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4
    ))
])

# Load data with features
X, y = client.ml.load_features(
    dataset="titanic",
    include_generated=True,  # Include MDM-generated features
    exclude_columns=['Name', 'Ticket']  # Exclude non-predictive columns
)

# Train pipeline
pipeline.fit(X, y)

# Make predictions on test set
test_data = client.load_dataset("titanic", table="test")
predictions = pipeline.predict(test_data)

# Create submission
client.ml.create_submission(
    dataset="titanic",
    predictions=predictions,
    output_path="submission.csv"
)
```

### Feature Importance Analysis

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Get feature importance
def analyze_feature_importance(model, X, y, dataset_name):
    """Analyze and visualize feature importance."""
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Tree-based feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {dataset_name}")
        plt.bar(range(20), importances[indices])
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    
    # Permutation importance (model-agnostic)
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=42
    )
    
    # Sort and display
    perm_indices = perm_importance.importances_mean.argsort()[::-1][:20]
    
    print("\nTop 20 Features by Permutation Importance:")
    for idx in perm_indices:
        print(f"{feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f}")
    
    return perm_importance

# Analyze
importance = analyze_feature_importance(model, X_test, y_test, "titanic")
```

## PyTorch Integration

### Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class MDMTorchDataset(Dataset):
    """PyTorch Dataset wrapper for MDM datasets."""
    
    def __init__(self, dataset_name, client, table="train", transform=None):
        self.client = client
        self.transform = transform
        
        # Load data
        data = client.load_dataset(dataset_name, table=table)
        info = client.get_dataset_info(dataset_name)
        
        # Separate features and target
        if info.target_column and info.target_column in data.columns:
            self.X = data.drop(columns=[info.target_column] + info.id_columns)
            self.y = data[info.target_column]
        else:
            self.X = data.drop(columns=info.id_columns)
            self.y = None
            
        # Convert to tensors
        self.X = torch.FloatTensor(self.X.values)
        if self.y is not None:
            # Classification vs regression
            if info.problem_type == "classification":
                self.y = torch.LongTensor(self.y.values)
            else:
                self.y = torch.FloatTensor(self.y.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
            
        if self.y is not None:
            return x, self.y[idx]
        return x

# Usage
train_dataset = MDMTorchDataset("titanic", client, table="train")
test_dataset = MDMTorchDataset("titanic", client, table="test")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Neural Network Example

```python
class TabularNN(nn.Module):
    """Neural network for tabular data."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(TabularNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Initialize model
input_dim = train_dataset.X.shape[1]
model = TabularNN(
    input_dim=input_dim,
    hidden_dims=[128, 64, 32],
    output_dim=2,  # Binary classification
    dropout=0.3
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Training loop
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# Train model
for epoch in range(50):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    scheduler.step(train_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}")
```

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class MDMLightningModule(pl.LightningModule):
    """PyTorch Lightning module for MDM datasets."""
    
    def __init__(self, input_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TabularNN(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            output_dim=2
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Create data module
class MDMDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, client, batch_size=32):
        super().__init__()
        self.dataset_name = dataset_name
        self.client = client
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = MDMTorchDataset(
            self.dataset_name, self.client, "train"
        )
        # Split train into train/val
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Train with Lightning
data_module = MDMDataModule("titanic", client)
model = MDMLightningModule(input_dim=train_dataset.X.shape[1])

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(monitor='val_acc', mode='max')
    ]
)

trainer.fit(model, data_module)
```

## TensorFlow/Keras Integration

### tf.data Pipeline

```python
import tensorflow as tf
from tensorflow import keras

def create_tf_dataset(dataset_name, client, batch_size=32, shuffle=True):
    """Create TensorFlow dataset from MDM."""
    
    # Load data
    data = client.load_dataset(dataset_name)
    info = client.get_dataset_info(dataset_name)
    
    # Separate features and labels
    X = data.drop(columns=[info.target_column] + info.id_columns).values
    y = data[info.target_column].values
    
    # Create tf.data dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, X.shape[1]

# Create datasets
train_ds, input_dim = create_tf_dataset("titanic", client, batch_size=32)
val_ds, _ = create_tf_dataset("titanic_val", client, batch_size=32, shuffle=False)
```

### Keras Model

```python
# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Custom callbacks
class MDMCallback(keras.callbacks.Callback):
    """Custom callback for MDM integration."""
    
    def __init__(self, client, dataset_name):
        self.client = client
        self.dataset_name = dataset_name
        
    def on_epoch_end(self, epoch, logs=None):
        # Log metrics to MDM monitoring
        if logs:
            self.client.log_metrics(
                dataset=self.dataset_name,
                metrics={
                    'epoch': epoch,
                    'loss': logs.get('loss'),
                    'accuracy': logs.get('accuracy'),
                    'val_loss': logs.get('val_loss'),
                    'val_accuracy': logs.get('val_accuracy')
                }
            )

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5),
        MDMCallback(client, "titanic")
    ]
)
```

### TensorFlow Feature Columns

```python
# Use TensorFlow feature columns with MDM features
def create_feature_columns(dataset_info):
    """Create TensorFlow feature columns from MDM dataset."""
    feature_columns = []
    
    for col_name, col_info in dataset_info.column_info.items():
        if col_info['dtype'] == 'numeric':
            # Numeric features
            feature_columns.append(
                tf.feature_column.numeric_column(col_name)
            )
        elif col_info['dtype'] == 'categorical':
            # Categorical features
            vocabulary = col_info.get('unique_values', [])
            if len(vocabulary) < 100:  # One-hot for low cardinality
                categorical = tf.feature_column.categorical_column_with_vocabulary_list(
                    col_name, vocabulary
                )
                feature_columns.append(
                    tf.feature_column.indicator_column(categorical)
                )
            else:  # Embedding for high cardinality
                categorical = tf.feature_column.categorical_column_with_hash_bucket(
                    col_name, hash_bucket_size=1000
                )
                feature_columns.append(
                    tf.feature_column.embedding_column(categorical, dimension=16)
                )
                
    return feature_columns

# Create feature columns
info = client.get_dataset_info("titanic")
feature_columns = create_feature_columns(info)

# Use in model
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = keras.Sequential([
    feature_layer,
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

## XGBoost Integration

### Basic Usage

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load data
X_train, X_test, y_train, y_test = client.ml.load_train_test_split("titanic")

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Train
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=True
)

# Feature importance
xgb.plot_importance(model, max_num_features=20)
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 1000),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5)
}

# Random search
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

# Fit
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Save best model
best_model = random_search.best_estimator_
```

## LightGBM Integration

```python
import lightgbm as lgb

# Prepare data
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42
}

# Train
model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(50)]
)

# Feature importance
lgb.plot_importance(model, max_num_features=20)

# Predictions
predictions = model.predict(X_test, num_iteration=model.best_iteration)
```

## Cross-Validation Strategies

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

# Create stratified folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate
cv_results = cross_validate(
    model,
    X, y,
    cv=skf,
    scoring=['accuracy', 'roc_auc', 'f1'],
    return_train_score=True
)

# Display results
for metric in ['accuracy', 'roc_auc', 'f1']:
    train_scores = cv_results[f'train_{metric}']
    val_scores = cv_results[f'test_{metric}']
    print(f"{metric}:")
    print(f"  Train: {train_scores.mean():.3f} (+/- {train_scores.std() * 2:.3f})")
    print(f"  Val: {val_scores.mean():.3f} (+/- {val_scores.std() * 2:.3f})")
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(dataset_name, model, n_splits=5):
    """Time series cross-validation for MDM datasets."""
    
    # Load data sorted by time
    data = client.load_dataset(dataset_name)
    info = client.get_dataset_info(dataset_name)
    
    # Sort by time column
    if info.time_column:
        data = data.sort_values(info.time_column)
    
    X = data.drop(columns=[info.target_column] + info.id_columns)
    y = data[info.target_column]
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold + 1}: {score:.3f}")
        
    print(f"Average: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
    
    return scores

# Use
scores = time_series_cv("sales_forecast", model)
```

## Ensemble Methods

### Voting Ensemble

```python
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', xgb.XGBClassifier(n_estimators=100)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100))
    ],
    voting='soft'  # Use predicted probabilities
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Predictions
ensemble_pred = ensemble.predict_proba(X_test)[:, 1]
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', xgb.XGBClassifier(n_estimators=100)),
    ('nn', MLPClassifier(hidden_layer_sizes=(100, 50)))
]

# Meta model
meta_model = LogisticRegression()

# Stacking ensemble
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Use 5-fold CV to train meta model
)

stacking.fit(X_train, y_train)
```

## Model Deployment

### Save and Load Models

```python
import joblib
import json

def save_mdm_model(model, dataset_name, model_name, metadata=None):
    """Save model with MDM metadata."""
    
    # Get dataset info
    info = client.get_dataset_info(dataset_name)
    
    # Prepare metadata
    model_metadata = {
        'dataset': dataset_name,
        'model_name': model_name,
        'features': info.columns['original'] + info.columns['features'],
        'target': info.target_column,
        'problem_type': info.problem_type,
        'created_at': datetime.now().isoformat(),
        'mdm_version': client.version,
        'custom_metadata': metadata or {}
    }
    
    # Save model
    model_path = f"{dataset_name}_{model_name}_model.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata_path = f"{dataset_name}_{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    
    return model_path, metadata_path

# Save
save_mdm_model(model, "titanic", "rf_v1", {
    'accuracy': 0.85,
    'hyperparameters': model.get_params()
})
```

### Model Serving

```python
class MDMModelServer:
    """Simple model server for MDM models."""
    
    def __init__(self, model_path, metadata_path):
        self.model = joblib.load(model_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
            
        self.client = MDMClient()
        
    def predict(self, data):
        """Make predictions on new data."""
        
        # Ensure correct features
        expected_features = self.metadata['features']
        
        # Add missing features with default values
        for feat in expected_features:
            if feat not in data.columns:
                data[feat] = 0  # Or appropriate default
                
        # Select and order features
        data = data[expected_features]
        
        # Make predictions
        predictions = self.model.predict(data)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(data)
            return predictions, probabilities
            
        return predictions
    
    def predict_from_raw(self, raw_data_path):
        """Predict from raw data file."""
        
        # Register temporary dataset
        temp_name = f"temp_{uuid.uuid4().hex[:8]}"
        self.client.register_dataset(
            temp_name,
            raw_data_path,
            no_features=False  # Generate features
        )
        
        # Load with features
        data = self.client.load_dataset(temp_name)
        
        # Predict
        predictions = self.predict(data)
        
        # Cleanup
        self.client.remove_dataset(temp_name, force=True)
        
        return predictions

# Use server
server = MDMModelServer("titanic_rf_v1_model.pkl", "titanic_rf_v1_metadata.json")
predictions = server.predict_from_raw("new_passengers.csv")
```

## MLflow Integration

```python
import mlflow
import mlflow.sklearn

# Configure MLflow with MDM
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("mdm_experiments")

def train_with_mlflow(dataset_name, model, params):
    """Train model with MLflow tracking."""
    
    with mlflow.start_run(run_name=f"{dataset_name}_{model.__class__.__name__}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log dataset info
        info = client.get_dataset_info(dataset_name)
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("dataset_rows", info.row_count)
        mlflow.log_param("dataset_features", len(info.columns['features']))
        
        # Load data
        X_train, X_test, y_train, y_test = client.ml.load_train_test_split(
            dataset_name
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"mdm_{dataset_name}_model"
        )
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save as artifact
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
        
        return model

# Train with tracking
model = train_with_mlflow(
    "titanic",
    RandomForestClassifier(),
    {'n_estimators': 100, 'max_depth': 10}
)
```

## Best Practices

### 1. Feature Engineering Pipeline

```python
def create_ml_pipeline(dataset_name):
    """Create complete ML pipeline with MDM features."""
    
    # Load data info
    info = client.get_dataset_info(dataset_name)
    
    # Identify feature types
    numeric_features = [col for col in info.columns['original'] 
                       if info.column_types[col] == 'numeric']
    categorical_features = [col for col in info.columns['original']
                           if info.column_types[col] == 'categorical']
    
    # Create preprocessors
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    return pipeline
```

### 2. Experiment Tracking

```python
class MDMExperiment:
    """Experiment tracking for MDM."""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.results = []
        
    def run_experiment(self, model, params, cv=5):
        """Run and track experiment."""
        
        # Load data
        X, y = client.ml.load_features(self.dataset_name)
        
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Record results
        result = {
            'model': model.__class__.__name__,
            'params': params,
            'cv_scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'timestamp': datetime.now()
        }
        
        self.results.append(result)
        
        return result
    
    def get_best_model(self):
        """Get best performing model."""
        return max(self.results, key=lambda x: x['mean_score'])
    
    def save_results(self, path):
        """Save experiment results."""
        pd.DataFrame(self.results).to_csv(path, index=False)

# Run experiments
experiment = MDMExperiment("titanic")

# Try different models
models = [
    (RandomForestClassifier(), {'n_estimators': 100}),
    (XGBClassifier(), {'n_estimators': 100}),
    (LogisticRegression(), {'C': 1.0})
]

for model, params in models:
    model.set_params(**params)
    experiment.run_experiment(model, params)

best = experiment.get_best_model()
print(f"Best model: {best['model']} with score {best['mean_score']:.3f}")
```

### 3. Competition Workflow

```python
def competition_workflow(dataset_name):
    """Complete competition workflow with MDM."""
    
    # 1. Load and explore
    train_df = client.load_dataset(dataset_name, table="train")
    test_df = client.load_dataset(dataset_name, table="test")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # 2. Feature engineering (already done by MDM)
    info = client.get_dataset_info(dataset_name)
    print(f"Generated features: {len(info.columns['features'])}")
    
    # 3. Prepare data
    X = train_df.drop(columns=[info.target_column] + info.id_columns)
    y = train_df[info.target_column]
    X_test = test_df.drop(columns=info.id_columns)
    
    # 4. Train model
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(X, y)
    
    # 5. Make predictions
    predictions = model.predict(X_test)
    
    # 6. Create submission
    submission = pd.DataFrame({
        info.id_columns[0]: test_df[info.id_columns[0]],
        info.target_column: predictions
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Submission created!")
    
    return model, submission

# Run workflow
model, submission = competition_workflow("titanic")
```