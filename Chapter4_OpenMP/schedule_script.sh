#!/bin/bash
# First parameter should be program to time
# Second parameter should be file to save timings to

for scheme in static dynamic guided
do
   for chunk in 1 2 4 8 16 32 
      do
        export OMP_SCHEDULE="${scheme},${chunk}"
        echo $OMP_SCHEDULE `/usr/bin/time -o tmp.log -p $1 >/dev/null ; head -n 1 tmp.log | gawk '{print $2}' ` >> $2
      done
done

