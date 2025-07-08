#!/usr/bin/env python3
"""Debug time series registration and analysis."""

import os
import tempfile
from pathlib import Path
import pandas as pd
import yaml
from typer.testing import CliRunner
from mdm.cli.main import app

# Create temp dir
with tempfile.TemporaryDirectory() as tmpdir:
    os.environ['MDM_HOME_DIR'] = tmpdir
    
    # Setup MDM structure
    mdm_path = Path(tmpdir)
    (mdm_path / "datasets").mkdir(parents=True)
    (mdm_path / "config" / "datasets").mkdir(parents=True)
    (mdm_path / "logs").mkdir(parents=True)
    
    # Create config
    config_file = mdm_path / "mdm.yaml"
    config_file.write_text("""
database:
  default_backend: sqlite
  sqlite:
    synchronous: NORMAL
    journal_mode: WAL
    
performance:
  batch_size: 10000
  
logging:
  level: DEBUG
  file: mdm.log
""")
    
    # Create time series data
    ts_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'value': [i + (i % 7) * 10 for i in range(100)],
        'id': range(1, 101)
    })
    
    ts_path = mdm_path / "timeseries.csv"
    ts_df.to_csv(ts_path, index=False)
    
    # Register dataset
    runner = CliRunner()
    result = runner.invoke(app, [
        "dataset", "register", "ts_test", str(ts_path),
        "--target", "value",
        "--time-column", "date",
        "--id-columns", "id"
    ])
    print("Registration result:")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.stdout}")
    
    # Check what was saved
    yaml_file = mdm_path / "config" / "datasets" / "ts_test.yaml"
    if yaml_file.exists():
        print("\nSaved YAML:")
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            print(f"time_column: {data.get('time_column')}")
            print(f"Full data: {data}")
    
    # Try to analyze
    result = runner.invoke(app, ["timeseries", "analyze", "ts_test"])
    print("\nAnalysis result:")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.stdout}")