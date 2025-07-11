#!/bin/bash
# Documentation reorganization script for MDM

echo "Starting documentation reorganization..."

# Ensure we're in the docs directory
cd "$(dirname "$0")"

# Create directory structure
echo "Creating directory structure..."
mkdir -p current/user current/api current/development
mkdir -p outdated/to-update outdated/no-longer-needed
mkdir -p archive/refactoring archive/migration-summaries

# CURRENT - User Documentation
echo "Moving user documentation to current/user/..."
for file in 00_Table_of_Contents.md 01_Project_Overview.md 02_Configuration.md \
            03_Database_Architecture.md 04_Dataset_Registration.md \
            05_Dataset_Management_Operations.md 06_Database_Backends.md \
            07_Command_Line_Interface.md 08_Programmatic_API.md \
            09_Advanced_Features.md 10_Best_Practices.md 11_Troubleshooting.md \
            12_Summary.md 13_Testing_and_Validation.md 14_Target_ID_Detection_Schema.md; do
    if [ -f "$file" ]; then
        mv "$file" current/user/
    fi
done

# Move tutorials if exists
if [ -d "tutorials" ]; then
    mv tutorials current/user/
fi

# CURRENT - API & Developer Guides
echo "Moving API documentation to current/api/..."
for file in API_Reference.md Developer_Guide.md Architecture_Design.md; do
    if [ -f "$file" ]; then
        mv "$file" current/api/
    fi
done

# CURRENT - Development Documentation
echo "Moving development documentation to current/development/..."
for file in Contributing.md Deployment_Guide.md MANUAL_TEST_CHECKLIST.md \
            Test_Fix_Patterns.md Performance_Optimization.md Monitoring_Guide.md; do
    if [ -f "$file" ]; then
        mv "$file" current/development/
    fi
done

# OUTDATED - Needs Update
echo "Moving outdated documentation that needs updating..."
for file in Migration_Guide.md README.md mdm.yaml.default \
            Troubleshooting.md Troubleshooting_Guide.md; do
    if [ -f "$file" ]; then
        mv "$file" outdated/to-update/
    fi
done

# ARCHIVE - Refactoring Process
echo "Archiving refactoring documentation..."
if [ -d "refactoring" ]; then
    mv refactoring archive/
fi
for file in Architecture_Decisions.md Refactoring_Safe_Migration_Guide.md \
            mdm-initial-prompt.md; do
    if [ -f "$file" ]; then
        mv "$file" archive/refactoring/
    fi
done

# ARCHIVE - Migration Summaries
echo "Archiving migration summaries..."
for file in CLI_Migration_Summary.md Configuration_Migration_Summary.md \
            Dataset_Registration_Migration_Summary.md Feature_Engineering_Migration_Summary.md \
            Integration_Testing_Summary.md Storage_Backend_Migration_Summary.md \
            Performance_Optimization_Summary.md Documentation_Summary.md \
            Rollout_Summary.md RELEASE_NOTES_UPDATE_IMPROVEMENTS.md; do
    if [ -f "$file" ]; then
        mv "$file" archive/migration-summaries/
    fi
done

# Create README files for each directory
echo "Creating README files..."

cat > current/README.md << 'EOF'
# Current Documentation / Aktualna Dokumentacja

This directory contains the current, up-to-date documentation for MDM after the 2025 refactoring.
Ten katalog zawiera aktualną dokumentację MDM po refaktoringu 2025.

## Structure / Struktura
- `user/` - User-facing documentation (guides, tutorials, CLI reference) / Dokumentacja użytkownika
- `api/` - API reference and architecture documentation / Dokumentacja API i architektury
- `development/` - Development guides (contributing, testing, deployment) / Przewodniki dla deweloperów

All documentation in this directory is current and accurate for the refactored MDM system.
Cała dokumentacja w tym katalogu jest aktualna dla zrefaktoryzowanego systemu MDM.
EOF

cat > current/user/README.md << 'EOF'
# User Documentation / Dokumentacja Użytkownika

Complete user documentation for MDM, organized in numbered sequence:
Kompletna dokumentacja użytkownika MDM, zorganizowana w numerowanej sekwencji:

- 00-14: Core documentation chapters / Główne rozdziały dokumentacji
- tutorials/: Step-by-step tutorials for common tasks / Samouczki krok po kroku

Start with `00_Table_of_Contents.md` for a complete overview.
Zacznij od `00_Table_of_Contents.md` dla pełnego przeglądu.
EOF

cat > outdated/README.md << 'EOF'
# Outdated Documentation / Nieaktualna Dokumentacja

This directory contains documentation that needs attention:
Ten katalog zawiera dokumentację wymagającą uwagi:

- `to-update/` - Important documentation that needs updating / Ważna dokumentacja do aktualizacji
- `no-longer-needed/` - Documentation obsolete after refactoring / Dokumentacja niepotrzebna po refaktoringu

Review and update documents in `to-update/` before moving to `current/`.
Przejrzyj i zaktualizuj dokumenty w `to-update/` przed przeniesieniem do `current/`.
EOF

cat > archive/README.md << 'EOF'
# Archive - Historical Documentation / Archiwum - Dokumentacja Historyczna

This directory preserves historical documentation from the refactoring process.
Ten katalog przechowuje historyczną dokumentację z procesu refaktoringu.

- `refactoring/` - Complete refactoring process documentation / Kompletna dokumentacja procesu refaktoringu
- `migration-summaries/` - Step-by-step migration summaries / Podsumowania migracji krok po kroku

These documents are kept for historical reference and understanding the evolution of MDM.
Te dokumenty są przechowywane jako odniesienie historyczne i dla zrozumienia ewolucji MDM.
EOF

echo ""
echo "Documentation reorganization complete!"
echo ""
echo "Summary / Podsumowanie:"
echo "- Current docs / Aktualne: $(find current -name "*.md" 2>/dev/null | wc -l) files"
echo "- Outdated to update / Do aktualizacji: $(find outdated/to-update -name "*.md" 2>/dev/null | wc -l) files"
echo "- Archived / Zarchiwizowane: $(find archive -name "*.md" 2>/dev/null | wc -l) files"
echo ""
echo "Next steps / Następne kroki:"
echo "1. Review files in outdated/to-update/ / Przejrzyj pliki w outdated/to-update/"
echo "2. Update cross-references to new paths / Zaktualizuj odniesienia do nowych ścieżek"
echo "3. Consolidate duplicate guides / Scal zduplikowane przewodniki"