#!/bin/sh

outdir=$1

sz=30
spd=3
puls=0.2

outfile="sample_$(printf %03d $sz)_$(printf %02d $spd).mp4"

python samplegen.py --size $sz --noise 5 --duration 10 --speed $spd --pulsation $puls -o $outdir/$outfile