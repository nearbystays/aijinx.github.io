#!/bin/bash

curl -otsla https://www.google.com/finance/quote/TSLA:NASDAQ
# printf "Current Time: " && date
printf "a: All After Hours\np: Spot Price (premarket)\nap: After Hours Price\n"
read -p "Choose: " choice
if [ "$choice" = "a" ]; then
    grep 'After Hours' tsla
elif [ "$choice" = "p" ]; then
    grep -o '\$[0-9]*\.[0-9]*' tsla | head -n 2 - | tail -n 1
elif [ "$choice" = "ap" ]; then
    grep 'After Hours' tsla | grep -o '\$[0-9]*\.[0-9]*' - | head -n 1 -
else
    cat tsla
fi
