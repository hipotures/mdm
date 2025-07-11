# ML Data Manager (MDM): Table of Contents

## Complete Documentation Guide

This documentation provides comprehensive information about the ML Data Manager (MDM) system - a standalone, enterprise-grade dataset management solution designed specifically for machine learning workflows.

### Documentation Structure

1. **[00_Table_of_Contents.md](00_Table_of_Contents.md)** - This file
   - Complete documentation guide and navigation

2. **[00_Quick_Start.md](00_Quick_Start.md)** - 5-Minute Quick Start Guide 🚀
   - Installation and basic usage
   - Common commands
   - Python API basics

3. **[01_Project_Overview.md](01_Project_Overview.md)** - Introduction and Overview
   - Project description
   - Language requirement (English only)
   - Key benefits
   - Use cases
   - System overview and capabilities

3. **[02_Configuration.md](02_Configuration.md)** - System Configuration
   - Main configuration (mdm.yaml)
   - Dataset configuration files
   - Storage paths and settings
   - Database backend configuration
   - Performance settings (batch_size, concurrency)

4. **[03_Database_Architecture.md](03_Database_Architecture.md)** - Database Design
   - Two-tier database system
   - Dataset discovery mechanism
   - Metadata storage
   - Performance considerations

5. **[04_Dataset_Registration.md](04_Dataset_Registration.md)** - Dataset Registration Process
   - Automatic registration (AUTO mode)
   - Manual registration (NO-AUTO mode)
   - Registration process details
   - Configuration file structure
   - Auto-detection logic
   - Registration examples

6. **[05_Dataset_Management_Operations.md](05_Dataset_Management_Operations.md)** - Core Operations
   - Register - Adding new datasets
   - List - Browse available datasets
   - Info - Detailed dataset information
   - Search - Find datasets
   - Export - Export datasets
   - Stats - Dataset statistics
   - Update - Modify dataset metadata
   - Remove - Delete datasets

7. **[06_Database_Backends.md](06_Database_Backends.md)** - Storage Backends
   - DuckDB (recommended default)
   - SQLite
   - PostgreSQL
   - Backend selection and configuration

8. **[07_Command_Line_Interface.md](07_Command_Line_Interface.md)** - CLI Reference
   - MDM commands
   - Registration options
   - Export/import options
   - Common options
   - Example usage scenarios

9. **[08_Programmatic_API.md](08_Programmatic_API.md)** - Python API
   - Dataset Manager API
   - Dataset Service API (Advanced)
   - Code examples

10. **[09_Advanced_Features.md](09_Advanced_Features.md)** - Advanced Functionality
    - Feature Engineering System
    - Performance optimization and batch processing
    - Memory management with configurable batch sizes
    - Integration features
    - Data type handling in metadata
    - Advanced usage patterns

11. **[10_Best_Practices.md](10_Best_Practices.md)** - Recommendations
    - Language standards (English only)
    - Dataset naming conventions
    - File organization
    - Column naming
    - Performance tips and batch size tuning
    - Security considerations
    - Maintenance procedures

12. **[11_Troubleshooting.md](11_Troubleshooting.md)** - Problem Solving
    - Common issues and solutions
    - Debug mode
    - Error messages
    - Performance problems and batch processing issues
    - Progress bar interpretation

13. **[12_Summary.md](12_Summary.md)** - Project Summary
    - Key principles
    - Architecture benefits
    - Ideal use cases
    - Final thoughts

14. **[13_Testing_and_Validation.md](13_Testing_and_Validation.md)** - Testing Strategies
    - End-to-end testing scripts
    - Available test scripts and their purposes
    - Creating custom test scripts
    - Continuous integration
    - Unit and integration testing

15. **[14_Target_ID_Detection_Schema.md](14_Target_ID_Detection_Schema.md)** - Target & ID Detection Logic
    - Complete detection flow diagram
    - Auto-detection rules
    - Manual mode requirements
    - Error scenarios and messages
    - Priority and override behavior

16. **[15_FAQ.md](15_FAQ.md)** - Frequently Asked Questions
    - General questions
    - Installation and setup
    - Common issues and solutions
    - Best practices

17. **[16_Environment_Variables.md](16_Environment_Variables.md)** - Environment Variables Reference
    - Complete list of all MDM_* variables
    - Configuration hierarchy
    - Usage examples
    - Common patterns

### Quick Start

For a quick introduction to MDM, start with:
1. **[00_Quick_Start.md](00_Quick_Start.md)** - Get started in 5 minutes! 🚀
2. [01_Project_Overview.md](01_Project_Overview.md) - Understand what MDM is
3. [04_Dataset_Registration.md](04_Dataset_Registration.md) - Learn dataset registration details
4. [15_FAQ.md](15_FAQ.md) - Find answers to common questions

### Navigation Tips

- Each document is self-contained and focuses on a specific aspect of MDM
- Cross-references between documents are provided where relevant
- Code examples and command snippets are included throughout
- For implementation details, refer to the source code in the parent directory