@echo off

FOR %%G IN (*) DO (
    REM Set filename without delayed expansion, no misinterpreted '!' possible
    set "old_filename=%%~G"
    REM before using old_filename, enable delayed expansion
    SetLocal EnableDelayedExpansion
    set "new_filename=!old_filename:-=!"
    REM Rename only if there is a difference between new filename and old filename in %%G
    ren "!old_filename!" "!new_filename!"
    REM No more delayed expansion needed in block now
    EndLocal
)

exit /b 0