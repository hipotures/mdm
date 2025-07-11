# Code Complexity Analysis Report

## Summary

This report analyzes the code complexity of key MDM modules using McCabe cyclomatic complexity standards:
- **1-10**: Simple, low risk ✅
- **11-20**: Moderate complexity ⚠️
- **21-50**: High complexity, refactoring candidate 🔴
- **50+**: Very high risk, must refactor ⚠️⚠️

## 1. src/mdm/dataset/registrar.py

### Class: DatasetRegistrar

#### Method: `register()` (lines 49-212)
- **Cyclomatic Complexity**: ~25-30 🔴
- **Lines of Code**: 163
- **Parameters**: 7
- **Max Nesting Depth**: 3
- **Code Smells**:
  - Long method (163 lines)
  - High complexity due to multiple conditional branches
  - Mixed responsibilities (validation, loading, feature generation)
  - Complex error handling with nested try-except blocks

#### Method: `_load_data_files()` (lines 335-669)
- **Cyclomatic Complexity**: ~45-50 🔴
- **Lines of Code**: 334
- **Parameters**: 4
- **Max Nesting Depth**: 5
- **Code Smells**:
  - God method (334 lines!)
  - Multiple file format handling in single method
  - Deeply nested control structures
  - Duplicate code patterns for different file formats
  - Mixed concerns (file reading, progress tracking, batch processing)

#### Method: `_detect_column_types_with_profiling()` (lines 1032-1208)
- **Cyclomatic Complexity**: ~35 🔴
- **Lines of Code**: 176
- **Parameters**: 7
- **Max Nesting Depth**: 4
- **Code Smells**:
  - Long method with multiple responsibilities
  - Complex conditional logic for type detection
  - Duplicate code between profiling paths

#### Method: `_compute_initial_statistics()` (lines 1258-1400)
- **Cyclomatic Complexity**: ~20 ⚠️
- **Lines of Code**: 142
- **Parameters**: 3
- **Max Nesting Depth**: 4
- **Code Smells**:
  - Long method
  - Complex progress tracking logic mixed with computation

### Overall Assessment: DatasetRegistrar
- **Critical Issues**: 
  - `_load_data_files()` is a massive god method that urgently needs refactoring
  - `register()` method violates Single Responsibility Principle
- **Recommendation**: High priority refactoring needed

## 2. src/mdm/api.py

### Class: MDMClient

#### Method: `register_dataset()` (lines 36-86)
- **Cyclomatic Complexity**: 2 ✅
- **Lines of Code**: 50
- **Parameters**: 11
- **Max Nesting Depth**: 1
- **Code Smells**:
  - High parameter count (11)
  - Otherwise clean delegation pattern

#### Method: `list_datasets()` (lines 102-146)
- **Cyclomatic Complexity**: 8 ✅
- **Lines of Code**: 44
- **Parameters**: 5
- **Max Nesting Depth**: 2
- **Code Smells**: None - well-structured

#### Method: `get_statistics()` (lines 384-435)
- **Cyclomatic Complexity**: 10 ✅
- **Lines of Code**: 51
- **Parameters**: 3
- **Max Nesting Depth**: 3
- **Code Smells**:
  - Complex fallback logic could be simplified

#### Method: `create_time_series_splits()` (lines 703-735)
- **Cyclomatic Complexity**: 6 ✅
- **Lines of Code**: 32
- **Parameters**: 6
- **Max Nesting Depth**: 2
- **Code Smells**: None

### Overall Assessment: MDMClient
- **Status**: Well-designed API class
- **Issues**: Minor - high parameter counts in some methods
- **Recommendation**: Low priority - consider parameter objects for methods with many parameters

## 3. src/mdm/features/generator.py

### Class: FeatureGenerator

#### Method: `generate_features()` (lines 27-101)
- **Cyclomatic Complexity**: 12 ⚠️
- **Lines of Code**: 74
- **Parameters**: 6
- **Max Nesting Depth**: 3
- **Code Smells**:
  - Moderate complexity but acceptable
  - Clear separation of generic and custom features

#### Method: `generate_feature_tables()` (lines 103-228)
- **Cyclomatic Complexity**: 15 ⚠️
- **Lines of Code**: 125
- **Parameters**: 8
- **Max Nesting Depth**: 4
- **Code Smells**:
  - Progress tracking mixed with business logic
  - High parameter count

#### Method: `_load_custom_features()` (lines 230-283)
- **Cyclomatic Complexity**: 10 ✅
- **Lines of Code**: 53
- **Parameters**: 2
- **Max Nesting Depth**: 4
- **Code Smells**:
  - Dynamic module loading complexity

### Overall Assessment: FeatureGenerator
- **Status**: Moderate complexity
- **Issues**: Acceptable complexity levels
- **Recommendation**: Low priority - consider extracting progress tracking

## 4. src/mdm/core/feature_flags.py

### Class: FeatureFlags

#### Method: `__init__()` (lines 48-56)
- **Cyclomatic Complexity**: 1 ✅
- **Lines of Code**: 8
- **Parameters**: 1
- **Max Nesting Depth**: 1
- **Code Smells**: None

#### Method: `_load_from_file()` (lines 58-76)
- **Cyclomatic Complexity**: 6 ✅
- **Lines of Code**: 18
- **Parameters**: 1
- **Max Nesting Depth**: 3
- **Code Smells**: None

#### Method: `is_enabled_for_user()` (lines 167-194)
- **Cyclomatic Complexity**: 7 ✅
- **Lines of Code**: 27
- **Parameters**: 3
- **Max Nesting Depth**: 2
- **Code Smells**: None - clean percentage rollout logic

### Overall Assessment: FeatureFlags
- **Status**: Excellent design
- **Issues**: None
- **Recommendation**: No refactoring needed

## 5. src/mdm/config/config.py

### Class: ConfigManager

#### Method: `_apply_environment_variables()` (lines 115-182)
- **Cyclomatic Complexity**: 25-30 🔴
- **Lines of Code**: 67
- **Parameters**: 2
- **Max Nesting Depth**: 4
- **Code Smells**:
  - Many hardcoded string manipulations
  - Complex conditional logic for parsing environment variables
  - Should be refactored with a mapping table

#### Method: `load()` (lines 32-68)
- **Cyclomatic Complexity**: 5 ✅
- **Lines of Code**: 36
- **Parameters**: 1
- **Max Nesting Depth**: 2
- **Code Smells**: None

#### Method: `_convert_env_value()` (lines 184-218)
- **Cyclomatic Complexity**: 8 ✅
- **Lines of Code**: 34
- **Parameters**: 2
- **Max Nesting Depth**: 3
- **Code Smells**: None - clean type conversion logic

### Overall Assessment: ConfigManager
- **Status**: Generally good, one problematic method
- **Issues**: `_apply_environment_variables()` needs refactoring
- **Recommendation**: Medium priority - refactor environment variable parsing

## Refactoring Priorities

### High Priority 🔴
1. **DatasetRegistrar._load_data_files()** - Extract file format handlers into separate methods/classes
2. **DatasetRegistrar.register()** - Break down into smaller, focused methods

### Medium Priority ⚠️
1. **ConfigManager._apply_environment_variables()** - Use configuration mapping instead of hardcoded conditionals
2. **DatasetRegistrar._detect_column_types_with_profiling()** - Simplify branching logic

### Low Priority ✅
1. **MDMClient** - Consider parameter objects for methods with many parameters
2. **FeatureGenerator.generate_feature_tables()** - Extract progress tracking

## Recommendations

1. **Apply Extract Method refactoring** to break down large methods
2. **Use Strategy Pattern** for file format handling in DatasetRegistrar
3. **Implement Builder Pattern** for complex object construction
4. **Extract progress tracking** into a decorator or separate concern
5. **Create parameter objects** for methods with >5 parameters