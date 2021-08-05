#!/bin/bash -f
##


fa='cdfmatchA_ens20_time2_fit.png'
fb='cdfmatchA_ens20_time2_corrected.png'
fcc='cdfmatchA_ens20_time14_fit.png'
fd='cdfmatchA_ens20_time14_corrected.png'
fe='cdfmatch_slope_A.png'
ff='cdfmatch_intercept_A.png'

cp ../$fa ./
cp ../$fb ./
cp ../$fcc ./
cp ../$fd ./
cp ../$fe ./
cp ../$ff ./

pntsz=20
pntsz2=30
lfh=+110
lfh=-200
cen=60

grav=north

convert $fa -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(a)' \
        fa.png

convert $fb -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(b)' \
        fb.png

convert $fcc -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(c)' \
        fc.png

convert $fd -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(d)' \
        fd.png

convert $fe -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(e)' \
        fe.png

convert $ff -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(f)' \
        ff.png
#--------------------------------------------------------------
#--------------------------------------------------------------


montage -trim fa.png \
        -trim fb.png \
        -trim fc.png \
        -trim fd.png\
        -trim fe.png\
        -trim ff.png\
        -geometry 900x400 -tile 2x3 cdfmatch.png

display cdfmatch.png

#display stats.png
