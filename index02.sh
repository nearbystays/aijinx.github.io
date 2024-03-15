#!/bin/bash

curl -s -otsla https://www.google.com/finance/quote/TSLA:NASDAQ
grep -o '\$[0-9]*\.[0-9]*' tsla | head -n 2 - | tail -n 1

