@echo off

REM Explicitly define the language directories
set "languages=arabic english spanish"

REM Loop through each specified language directory
for %%L in (%languages%) do (
    REM Change to the language directory
    pushd %%L
    echo Processing language directory: %%L
    
    REM Loop through train and test subdirectories
    for %%T in (train test) do (
        REM Create output directory for converted files
        mkdir %%L\%%T\converted
        
        REM Loop through each .conllu file in the current subdirectory (train or test)
        echo Processing %%T subdirectory in %%L
        for %%F in (%%L\%%T\*.conllu) do (
            REM Convert the .conllu file to .spacy format
            echo Converting %%F to .spacy format
            python -m spacy convert "%%F" "%%L\%%T\converted" -n 10 --converter conllu
        )
    )

    REM Call the training function in the language directory
    echo Running spacy training for %%L
    python -m spacy train config.cfg --output ./output --gpu-id 0

    REM Call the evaluation function in the language directory
    echo Running spacy evaluation for %%L
    python -m spacy evaluate ./output/model_best %%L\test\ --gpu-id 0
    
    REM Return to the base directory
    popd
)

echo Done processing all languages.
pause
