#!/bin/sh

outdir=$1

for sz in 10 20 30 50 100 150
do
  for spd in 2 $(seq 5 5 60)
  do
    #echo $spd
    outfile="sample_$(printf %03d $sz)_$(printf %02d $spd).mp4"
    #echo $outfile
    python samplegen.py --size $sz --noise 5 --duration 60 --speed $spd -o $outdir/$outfile
  done
done
