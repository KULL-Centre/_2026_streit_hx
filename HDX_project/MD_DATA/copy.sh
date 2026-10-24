#!/bin/bash

SRC="../../../MD_DATA"

for DIR in "$SRC"/*/; do
    [ -d "$DIR" ] || continue

    NAME=$(basename "$DIR")
    mkdir -p "$NAME"

    for FILE in "$DIR"*.pdb "$DIR"*.sh; do
        [ -f "$FILE" ] || continue
        cp -p "$FILE" "$NAME/"
    done

    echo "Copied .pdb and .sh files to $NAME/"
done
