# MDM Architecture Diagrams

This file contains Mermaid diagrams used throughout the MDM documentation.

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI<br/>Typer + Rich]
        API[Python API<br/>MDMClient]
    end
    
    subgraph "Service Layer"
        DM[Dataset Manager]
        FG[Feature Generator]
        QE[Query Engine]
        EX[Export Service]
    end
    
    subgraph "Storage Layer"
        SF[Storage Factory]
        SQLite[SQLite Backend]
        DuckDB[DuckDB Backend]
        PG[PostgreSQL Backend]
    end
    
    subgraph "Data Layer"
        YML[YAML Configs<br/>~/.mdm/config/]
        DB1[Dataset DBs<br/>~/.mdm/datasets/]
    end
    
    CLI --> DM
    API --> DM
    API --> QE
    API --> EX
    
    DM --> SF
    DM --> FG
    QE --> SF
    EX --> SF
    
    SF --> SQLite
    SF --> DuckDB
    SF --> PG
    
    SQLite --> DB1
    DuckDB --> DB1
    PG --> DB1
    
    DM --> YML
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant DatasetManager
    participant FeatureGenerator
    participant StorageBackend
    participant Database
    
    User->>CLI: mdm dataset register sales data.csv
    CLI->>DatasetManager: register_dataset()
    DatasetManager->>DatasetManager: Validate name & path
    DatasetManager->>DatasetManager: Detect structure
    DatasetManager->>StorageBackend: create_backend()
    StorageBackend->>Database: CREATE TABLES
    DatasetManager->>DatasetManager: Load data in batches
    DatasetManager->>Database: INSERT data
    DatasetManager->>FeatureGenerator: generate_features()
    FeatureGenerator->>FeatureGenerator: Apply transformers
    FeatureGenerator->>Database: CREATE feature tables
    DatasetManager->>Database: Compute statistics
    DatasetManager->>CLI: Success
    CLI->>User: Dataset registered!
```

## Feature Engineering Pipeline

```mermaid
graph LR
    subgraph "Input"
        CSV[CSV File]
    end
    
    subgraph "Column Detection"
        DT{Detect Types}
        NUM[Numeric]
        CAT[Categorical]
        TMP[Temporal]
        TXT[Text]
    end
    
    subgraph "Generic Transformers"
        STAT[Statistical<br/>- zscore<br/>- log<br/>- outliers]
        CATF[Categorical<br/>- one-hot<br/>- frequency<br/>- target encoding]
        TMPF[Temporal<br/>- components<br/>- cyclical<br/>- is_weekend]
        TXTF[Text<br/>- length<br/>- word count<br/>- patterns]
    end
    
    subgraph "Custom"
        CUST[Custom Features<br/>dataset_name.py]
    end
    
    subgraph "Output"
        FT[Feature Table]
    end
    
    CSV --> DT
    DT --> NUM --> STAT
    DT --> CAT --> CATF
    DT --> TMP --> TMPF
    DT --> TXT --> TXTF
    
    STAT --> FT
    CATF --> FT
    TMPF --> FT
    TXTF --> FT
    CUST --> FT
```

## Storage Architecture

```mermaid
graph TB
    subgraph "MDM Storage System"
        subgraph "Discovery Layer"
            YAML[YAML Configs<br/>Lightweight Pointers]
        end
        
        subgraph "Storage Backends"
            direction LR
            SQLite[SQLite<br/>File-based<br/>Zero Config]
            DuckDB[DuckDB<br/>Columnar<br/>Analytics]
            PostgreSQL[PostgreSQL<br/>Client-Server<br/>Multi-user]
        end
        
        subgraph "Data Organization"
            DS1[dataset_1/<br/>├── data.db<br/>├── metadata.json<br/>└── features.db]
            DS2[dataset_2/<br/>├── data.db<br/>├── metadata.json<br/>└── features.db]
            DS3[dataset_n/<br/>├── data.db<br/>├── metadata.json<br/>└── features.db]
        end
    end
    
    YAML --> SQLite
    YAML --> DuckDB
    YAML --> PostgreSQL
    
    SQLite --> DS1
    DuckDB --> DS2
    PostgreSQL --> DS3
```

## Query Execution

```mermaid
graph TD
    subgraph "Query Types"
        Q1[Load Dataset]
        Q2[Filter Query]
        Q3[Aggregate Query]
        Q4[SQL Query]
    end
    
    subgraph "Query Engine"
        QP[Query Planner]
        QO[Query Optimizer]
        QE[Query Executor]
    end
    
    subgraph "Optimizations"
        IDX[Use Indexes]
        CACHE[Check Cache]
        BATCH[Batch Fetch]
    end
    
    subgraph "Results"
        DF[Pandas DataFrame]
        JSON[JSON Response]
        CSV[CSV Export]
    end
    
    Q1 --> QP
    Q2 --> QP
    Q3 --> QP
    Q4 --> QP
    
    QP --> QO
    QO --> IDX
    QO --> CACHE
    QO --> BATCH
    
    IDX --> QE
    CACHE --> QE
    BATCH --> QE
    
    QE --> DF
    QE --> JSON
    QE --> CSV
```

## ML Integration Flow

```mermaid
graph LR
    subgraph "MDM"
        DS[Dataset]
        FE[Features]
        META[Metadata]
    end
    
    subgraph "Frameworks"
        SK[Scikit-learn]
        PT[PyTorch]
        TF[TensorFlow]
        XGB[XGBoost]
    end
    
    subgraph "Workflow"
        LOAD[Load Data]
        SPLIT[Train/Test Split]
        TRAIN[Train Model]
        PRED[Predictions]
        SUB[Submission]
    end
    
    DS --> LOAD
    FE --> LOAD
    META --> LOAD
    
    LOAD --> SK
    LOAD --> PT
    LOAD --> TF
    LOAD --> XGB
    
    SK --> SPLIT
    PT --> SPLIT
    TF --> SPLIT
    XGB --> SPLIT
    
    SPLIT --> TRAIN
    TRAIN --> PRED
    PRED --> SUB
```

## Configuration Hierarchy

```mermaid
graph TD
    subgraph "Priority (Highest to Lowest)"
        ENV[Environment Variables<br/>MDM_DATABASE_DEFAULT_BACKEND]
        CONF[Config File<br/>~/.mdm/mdm.yaml]
        DEF[Default Values<br/>Built-in]
    end
    
    ENV --> CONF
    CONF --> DEF
    
    subgraph "Resolution"
        CHECK{Value Set?}
        USE[Use Value]
        NEXT[Check Next Level]
    end
    
    ENV --> CHECK
    CHECK -->|Yes| USE
    CHECK -->|No| NEXT
    NEXT --> CONF
```

## Performance Optimization

```mermaid
graph TB
    subgraph "Bottlenecks"
        IO[I/O Bound]
        CPU[CPU Bound]
        MEM[Memory Bound]
    end
    
    subgraph "Solutions"
        subgraph "I/O Optimizations"
            BATCH[Larger Batches]
            SSD[Use SSD]
            COMP[Compression]
        end
        
        subgraph "CPU Optimizations"
            PARALLEL[Parallel Processing]
            VECTOR[Vectorization]
            INDEX[Indexing]
        end
        
        subgraph "Memory Optimizations"
            CHUNK[Chunking]
            DTYPE[Optimize Dtypes]
            STREAM[Streaming]
        end
    end
    
    IO --> BATCH
    IO --> SSD
    IO --> COMP
    
    CPU --> PARALLEL
    CPU --> VECTOR
    CPU --> INDEX
    
    MEM --> CHUNK
    MEM --> DTYPE
    MEM --> STREAM
```

## Dataset Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Raw: Data File
    Raw --> Validating: mdm register
    Validating --> Loading: Validation OK
    Validating --> [*]: Validation Failed
    Loading --> TypeDetection: Load Batches
    TypeDetection --> FeatureGen: Detect Types
    FeatureGen --> Statistics: Generate Features
    Statistics --> Ready: Compute Stats
    Ready --> InUse: Dataset Active
    InUse --> Updated: mdm update
    Updated --> Ready: Re-process
    InUse --> Exported: mdm export
    InUse --> Removed: mdm remove
    Removed --> [*]: Cleanup
```

## Error Handling Flow

```mermaid
graph TD
    subgraph "Error Types"
        DE[Dataset Error]
        BE[Backend Error]
        VE[Validation Error]
        CE[Config Error]
    end
    
    subgraph "Handling"
        LOG[Log Error]
        RETRY[Retry Logic]
        FALLBACK[Fallback Option]
        USER[User Message]
    end
    
    subgraph "Recovery"
        FIX[Auto-fix]
        SUGGEST[Suggest Fix]
        ABORT[Clean Abort]
    end
    
    DE --> LOG
    BE --> LOG
    VE --> LOG
    CE --> LOG
    
    LOG --> RETRY
    RETRY -->|Success| FIX
    RETRY -->|Fail| FALLBACK
    FALLBACK -->|Available| FIX
    FALLBACK -->|None| SUGGEST
    SUGGEST --> USER
    USER --> ABORT
```

## Backend Selection

```mermaid
graph TB
    subgraph "Single User Setup"
        USER[User]
        MDM[MDM CLI/API]
        CONFIG[~/.mdm/mdm.yaml]
    end
    
    subgraph "Backend Options"
        SQLite[SQLite<br/>Default<br/>Zero Config]
        DuckDB[DuckDB<br/>Analytics<br/>Fast Queries]
        PostgreSQL[PostgreSQL<br/>Advanced<br/>Remote DB]
    end
    
    USER --> MDM
    MDM --> CONFIG
    CONFIG -->|default_backend| SQLite
    CONFIG -->|default_backend| DuckDB
    CONFIG -->|default_backend| PostgreSQL
    
    subgraph "Use Cases"
        UC1[Personal Use<br/>→ SQLite]
        UC2[Heavy Analytics<br/>→ DuckDB]
        UC3[Team Shared DB<br/>→ PostgreSQL]
    end
```

These diagrams can be rendered in any Markdown viewer that supports Mermaid (GitHub, GitLab, many documentation tools).