#!/bin/bash -f
##

fcc='awdir_GEFS_vs_GFS0p25.png'
fd='wspd_GEFS_vs_GFS0p25.png'
fa='TPP6_GEFS_vs_GFS0p25.png'
fb='PBLH_GEFS_vs_GFS0p25.png'
fe='TEMP_GEFS_vs_GFS0p25.png'

cp ../$fa ./
cp ../$fb ./
cp ../$fcc ./
cp ../$fd ./
cp ../$fe ./
#cp ../$ff ./

pntsz=20
pntsz2=30
#lfh=-10
lfh=-120
lfhe=+120
cen=50

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
        -font Times-Roman -pointsize $pntsz -annotate $lfhe+$cen '(e)' \
        fe.png

montage -trim fa.png\
        -trim fb.png\
        -trim fc.png \
        -trim fd.png \
        -trim fe.png \
        -geometry 300x200 -tile 2x3 metdataA.png


display metdataA.png

