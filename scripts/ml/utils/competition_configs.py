"""Competition configurations for MDM generic features benchmark."""

from typing import Dict, Any

COMPETITIONS: Dict[str, Dict[str, Any]] = {
    'simple_benchmark': {
        'path': '/tmp/simple_benchmark',
        'target': 'target',
        'problem_type': 'binary_classification',
        'metric': 'accuracy',
        'id_column': 'id',
        'description': 'Simple synthetic benchmark dataset'
    },
    'titanic': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/Titanic',
        'target': 'Survived',
        'problem_type': 'binary_classification',
        'metric': 'accuracy',
        'id_column': 'PassengerId',
        'description': 'Titanic survival prediction'
    },
    'playground-s4e2': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e2',
        'target': 'NObeyesdad',
        'problem_type': 'multiclass_classification',
        'metric': 'accuracy',
        'id_column': 'id',
        'description': 'Obesity level classification'
    },
    'playground-s4e3': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e3',
        'target': ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'],
        'problem_type': 'multilabel_classification',
        'metric': 'roc_auc',
        'id_column': 'id',
        'description': 'Steel plate defects (multi-label)'
    },
    'playground-s4e4': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e4',
        'target': 'Rings',
        'problem_type': 'regression',
        'metric': 'rmse',
        'id_column': 'id',
        'description': 'Abalone age prediction'
    },
    'playground-s4e5': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e5',
        'target': 'FloodProbability',
        'problem_type': 'regression',
        'metric': 'mae',
        'id_column': 'id',
        'description': 'Flood probability prediction'
    },
    'playground-s4e6': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e6',
        'target': 'Target',
        'problem_type': 'multiclass_classification',
        'metric': 'accuracy',
        'id_column': 'id',
        'description': 'Student outcome prediction (Graduate/Dropout/Enrolled)'
    },
    'playground-s4e11': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e11',
        'target': 'Depression',
        'problem_type': 'binary_classification',
        'metric': 'accuracy',
        'id_column': 'id',
        'description': 'Depression detection'
    },
    'playground-s4e12': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s4e12',
        'target': 'Premium Amount',
        'problem_type': 'regression',
        'metric': 'rmse',
        'id_column': 'id',
        'description': 'Insurance premium prediction'
    },
    'playground-s4e10': {
        'path': '/mnt/ml/competitions/playground-series-s4e10',
        'target': 'loan_status',
        'problem_type': 'binary_classification',
        'metric': 'roc_auc',
        'id_column': 'id',
        'description': 'Loan default prediction'
    },
    'playground-s5e1': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s5e1/data',
        'target': 'num_sold',
        'problem_type': 'regression',
        'metric': 'rmse',
        'id_column': 'id',
        'description': 'Sales forecasting',
        'special_handling': 'time_series'
    },
    'playground-s5e6': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s5e6',
        'target': 'Fertilizer Name',
        'problem_type': 'multiclass_classification',
        'metric': 'accuracy',
        'id_column': 'id',
        'description': 'Fertilizer recommendation'
    },
    'playground-s5e7': {
        'path': '/home/xai/DEV/mdm/scripts/ml/competitions/playground-series-s5e7',
        'target': 'Personality',
        'problem_type': 'binary_classification',
        'metric': 'accuracy',
        'id_column': 'id',
        'description': 'Personality type prediction (Introvert/Extrovert)'
    }
}

def get_competition_config(name: str) -> Dict[str, Any]:
    """Get configuration for a specific competition."""
    if name not in COMPETITIONS:
        raise ValueError(f"Competition '{name}' not found. Available: {list(COMPETITIONS.keys())}")
    return COMPETITIONS[name]

def get_all_competitions() -> Dict[str, Dict[str, Any]]:
    """Get all competition configurations."""
    return COMPETITIONS

def get_competitions_by_type(problem_type: str) -> Dict[str, Dict[str, Any]]:
    """Get competitions filtered by problem type."""
    return {
        name: config 
        for name, config in COMPETITIONS.items() 
        if config['problem_type'] == problem_type
    }