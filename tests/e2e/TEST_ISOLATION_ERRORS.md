# Test Isolation Errors - Konkretne Przykłady

## 1. Dataset Already Exists - Przykłady z rzeczywistych testów

### Przykład 1: test_verify_yaml_settings_applied
```
FAILED tests/e2e/test_01_config/test_11_yaml.py::TestYAMLConfiguration::test_verify_yaml_settings_applied

subprocess.CalledProcessError: Command '['/home/xai/DEV2/mdm/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_yaml', '/tmp/mdm_test_cd26420c/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

Command failed: /home/xai/DEV2/mdm/.venv/bin/python -m mdm.cli.main dataset register test_yaml /tmp/mdm_test_cd26420c/test_data/sample_data.csv --target value
stdout: Error: Dataset 'test_yaml' already exists
```

**Problem**: Test używa nazwy `test_yaml`, która już istnieje z poprzedniego testu lub z manualnego testowania.

### Przykład 2: test_modify_yaml_changes_take_effect
```
FAILED tests/e2e/test_01_config/test_11_yaml.py::TestYAMLConfiguration::test_modify_yaml_changes_take_effect

subprocess.CalledProcessError: Command '['/home/xai/DEV2/mdm/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_duckdb', '/tmp/mdm_test_31d9c4bf/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

Command failed: /home/xai/DEV2/mdm/.venv/bin/python -m mdm.cli.main dataset register test_duckdb /tmp/mdm_test_31d9c4bf/test_data/sample_data.csv --target value
stdout: Error: Dataset 'test_duckdb' already exists
```

**Problem**: Nazwa `test_duckdb` koliduje z innym testem lub pozostałościami z poprzednich uruchomień.

### Przykład 3: test_mdm_log_level_debug
```
FAILED tests/e2e/test_01_config/test_12_env.py::TestEnvironmentVariables::test_mdm_log_level_debug

subprocess.CalledProcessError: Command '['/home/xai/DEV2/mdm/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_debug', '/tmp/mdm_test_911a3a82/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

Command failed: /home/xai/DEV2/mdm/.venv/bin/python -m mdm.cli.main dataset register test_debug /tmp/mdm_test_911a3a82/test_data/sample_data.csv --target value
stdout: Error: Dataset 'test_debug' already exists
```

### Przykład 4: test_unknown_configuration_keys
```
FAILED tests/e2e/test_01_config/test_11_yaml.py::TestYAMLConfiguration::test_unknown_configuration_keys

subprocess.CalledProcessError: Command '['/home/xai/DEV2/mdm/.venv/bin/python', '-m', 'mdm.cli.main', 'dataset', 'register', 'test_unknown_keys', '/tmp/mdm_test_31d9c4bf/test_data/sample_data.csv', '--target', 'value']' returned non-zero exit status 1.

Command failed: /home/xai/DEV2/mdm/.venv/bin/python -m mdm.cli.main dataset register test_unknown_keys /tmp/mdm_test_31d9c4bf/test_data/sample_data.csv --target value
stdout: Error: Dataset 'test_unknown_keys' already exists
```

## 2. Widoczność Datasetów z ~/.mdm

### Przykład z test_invalid_yaml_syntax
```
TestYAMLConfiguration.test_invalid_yaml_syntax
assert result.returncode != 0
E   assert 0 != 0

stdout: Registered Datasets
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Name      ┃ Type      ┃ Target    ┃ Tables ┃ Rows     ┃ MEM Size ┃ Backend   ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
│ clean_fi… │ multicla… │ target    │ 6      │ 15       │ 2.3 KB   │ sqlite    │
│ clean_te… │ multicla… │ target    │ 4      │ 15       │ 2.3 KB   │ sqlite    │
│ final_qu… │ multicla… │ target    │ 4      │ 15       │ 2.3 KB   │ sqlite    │
│ test_yaml │ regressi… │ value     │ 1      │ 100      │ 19.9 KB  │ sqlite    │
└───────────┴───────────┴───────────┴────────┴──────────┴──────────┴───────────┘

Warning: 4 dataset(s) use a different backend than the current 'sqlite' backend.
```

**Problem**: Test widzi datasety z głównego katalogu ~/.mdm zamiast z izolowanego środowiska testowego w /tmp.

## 3. Analiza Problemu

### Używane nazwy datasetów w testach:
```python
# Z test_11_yaml.py:
- "test_yaml"
- "test_duckdb" 
- "test_sqlite"
- "test_with_config"
- "test_unknown_keys"

# Z test_12_env.py:
- "test_debug"
- "test_sqlite_env"
- "test_duckdb_env"
- "test_batch"

# Z test_21_register.py:
- "test_single_csv"
- "test_dir_csv"
- "test_kaggle"
- "test_force"
```

### Problemy:
1. **Stałe nazwy**: Każdy test używa hardkodowanej nazwy datasetu
2. **Brak cleanup**: Datasety nie są usuwane po teście
3. **Współdzielony stan**: Testy w tej samej sesji widzą te same datasety
4. **Niepełna izolacja**: MDM_HOME_DIR nie izoluje wszystkich operacji

## 4. Rozwiązania

### Rozwiązanie 1: Unikalne nazwy
```python
import uuid
import time

def test_example(clean_mdm_env, run_mdm):
    # Zamiast: dataset_name = "test_yaml"
    dataset_name = f"test_yaml_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    # Wynik: "test_yaml_1736251200_a3f2b1"
```

### Rozwiązanie 2: Force flag
```python
def test_example(clean_mdm_env, run_mdm):
    result = run_mdm([
        "dataset", "register", "test_yaml", str(csv_file),
        "--target", "value",
        "--force"  # Nadpisz jeśli istnieje
    ])
```

### Rozwiązanie 3: Cleanup przed testem
```python
def test_example(clean_mdm_env, run_mdm):
    # Sprawdź czy dataset istnieje i usuń go
    list_result = run_mdm(["dataset", "list"])
    if "test_yaml" in list_result.stdout:
        run_mdm(["dataset", "remove", "test_yaml", "--force"])
    
    # Teraz rejestruj
    result = run_mdm(["dataset", "register", "test_yaml", ...])
```

### Rozwiązanie 4: Fixture z unikalną nazwą
```python
@pytest.fixture
def unique_dataset_name():
    """Generate unique dataset name for each test."""
    return f"test_{uuid.uuid4().hex[:8]}"

def test_example(clean_mdm_env, run_mdm, unique_dataset_name):
    result = run_mdm([
        "dataset", "register", unique_dataset_name, str(csv_file),
        "--target", "value"
    ])
```

## 5. Podsumowanie

Główny problem to używanie stałych nazw datasetów we wszystkich testach, co powoduje kolizje gdy:
- Testy są uruchamiane wielokrotnie
- Testy działają równolegle
- Pozostały datasety z manualnego testowania
- Izolacja środowiska nie jest pełna

Najlepsze rozwiązanie to kombinacja:
1. Unikalne nazwy datasetów
2. Właściwa izolacja środowiska (MDM_HOME_DIR)
3. Cleanup po każdym teście