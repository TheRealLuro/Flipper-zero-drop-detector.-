@echo off
setlocal enabledelayedexpansion
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo Failed to initialize MSVC environment
    exit /b 1
)

rem ---- Layout ----
rem   src/        : flip-keeper C++ + resources
rem   vendor/imgui: vendored Dear ImGui
rem   _build/     : intermediates (.obj, .res, .pdb)
rem   .           : final flip-keeper.exe + build_app.bat
set SRC=src
set IMGUI=vendor\imgui
set OUT=_build

if not exist %OUT% mkdir %OUT%

set SOURCES=%SRC%\app.cpp ^
    %IMGUI%\imgui.cpp %IMGUI%\imgui_draw.cpp %IMGUI%\imgui_tables.cpp %IMGUI%\imgui_widgets.cpp ^
    %IMGUI%\backends\imgui_impl_win32.cpp %IMGUI%\backends\imgui_impl_dx11.cpp

rem Compile the resource file (icon + version info). /i %SRC% lets the .rc find the .ico
rem next to it without relative-path gymnastics.
rc.exe /nologo /i %SRC% /fo %OUT%\flip_keeper.res %SRC%\flip_keeper.rc
if errorlevel 1 (
    echo Failed to compile %SRC%\flip_keeper.rc
    exit /b 1
)

rem Compile everything; redirect all .obj files into _build via /Fo.
cl.exe /nologo /std:c++20 /O2 /EHsc /MD /W3 /utf-8 ^
    /I%SRC% /I%IMGUI% /I%IMGUI%\backends ^
    /Fo%OUT%\ /Fe:flip-keeper.exe ^
    %SOURCES% %OUT%\flip_keeper.res ^
    /link /subsystem:windows d3d11.lib dxgi.lib dxguid.lib comdlg32.lib ole32.lib uuid.lib shell32.lib dwmapi.lib

exit /b %errorlevel%
