@echo off
REM Builds both source files with optimisation, then runs the quick smoke test
cl /nologo /std:c++17 /O2 /EHsc main.cpp algorithms.cpp /Fe:strassen.exe && strassen.exe quick