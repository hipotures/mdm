"""YDF file logging helper."""

import os
import sys
import contextlib

def train_with_file_logging(learner, train_data, valid_data=None, log_path=None, silent=True, verbose=0):
    """Train YDF model with output redirected to file."""
    
    if silent and log_path:
        # Redirect stdout/stderr to log file
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            with open(log_path, 'w', buffering=1) as log_file:
                sys.stdout = log_file
                sys.stderr = log_file
                
                # Train model
                if valid_data is not None:
                    model = learner.train(train_data, valid=valid_data, verbose=verbose)
                else:
                    model = learner.train(train_data, verbose=verbose)
                    
                return model, log_path
                
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    else:
        # Train normally
        if valid_data is not None:
            model = learner.train(train_data, valid=valid_data, verbose=verbose)
        else:
            model = learner.train(train_data, verbose=verbose)
            
        return model, log_path