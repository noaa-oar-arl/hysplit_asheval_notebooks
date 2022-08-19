#!/bin/bash -f
##


ens='gec00'
#fa=cdfmatchM_ens${ens}_time4_fit.png
#fb=cdfmatchM_ensgec00_time4_corrected.png
#fcc=cdfmatchM_ensgec00_time14_fit.png
#fd=cdfmatchM_ensgec00_time14_corrected.png
fe=cdfmatch_slope_M.png
ff=cdfmatch_intercept_M.png

fa=cdfmatch_pmeshM_t10_ens${ens}.png

cp ../$fa ./
#cp ../$fb ./
#cp ../$fcc ./
#cp ../$fd ./
cp ../$fe ./
cp ../$ff ./

pntsz=20
pntsz2=30
lfh=+110
lfh=-200
cen=60

grav=north

#convert $fa -gravity $grav\
#        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(a)' \
#        fa.png

#convert $fb -gravity $grav\
#        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(b)' \
#        fb.png

#convert $fcc -gravity $grav\
#        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(c)' \
#        fcc.png

#convert $fd -gravity $grav\
#        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(d)' \
#        fd.png

convert $fe -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '' \
        fe.png

convert $ff -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '' \
        ff.png
#--------------------------------------------------------------
#--------------------------------------------------------------

montage -trim fe.png \
        -trim ff.png\
        -geometry 600x400 -tile 2x1 feff.png

montage -trim $fa \
        -trim feff.png\
        -geometry 1000x400 -tile 1x2 cdfmatchM.png

display cdfmatchM.png

#display stats.png
