cl /nologo /std:c++17 /O2 /EHsc main.cpp algorithms.cpp /Fe:strassen.exe && strassen.exe
@echo off
REM Just for a faster compilation