
import sys
import pandas as pd
import ydf

# Redirect output to log
with open('ydf_logs/ydf_20250713_032611.log', 'w', buffering=1) as log_file:
    sys.stdout = log_file
    sys.stderr = log_file
    
    try:
        # Load data
        train_data = pd.read_pickle('ydf_logs/ydf_20250713_032611_train.pkl')
        valid_data = None
        
        # Recreate learner
        import json
        with open('ydf_logs/ydf_20250713_032611_learner.pkl', 'r') as f:
            config = json.load(f)
        
        # Recreate feature selector if needed
        feature_selector = None
        if 'feature_selector_config' in config['learner_params']:
            fs_config = config['learner_params']['feature_selector_config']
            feature_selector = ydf.BackwardSelectionFeatureSelector(
                removal_ratio=fs_config['removal_ratio'],
                objective_metric=fs_config['objective_metric'],
                maximize_objective=fs_config['maximize_objective']
            )
        
        # Create learner with all parameters
        params = config['learner_params'].copy()
        params.pop('feature_selector_config', None)  # Remove this, we handle it separately
        
        # Convert task string back to enum
        if 'task' in params and params['task']:
            if 'CLASSIFICATION' in params['task']:
                params['task'] = ydf.Task.CLASSIFICATION
            elif 'REGRESSION' in params['task']:
                params['task'] = ydf.Task.REGRESSION
        
        # Add feature selector
        if feature_selector:
            params['feature_selector'] = feature_selector
        
        # Create learner
        if config['learner_type'] == 'GradientBoostedTreesLearner':
            learner = ydf.GradientBoostedTreesLearner(**params)
        else:
            learner = ydf.RandomForestLearner(**params)
        
        # Train
        if valid_data is not None:
            model = learner.train(train_data, valid=valid_data)
        else:
            model = learner.train(train_data)
        
        # Save model
        model.save('ydf_logs/ydf_20250713_032611_model')
        
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
