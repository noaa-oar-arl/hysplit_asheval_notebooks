#!/bin/bash -f
##


fa='runB_CSI_ts_t0.1.png'
fb='runB_POD_ts_t0.1.png'
fcc='runB_FAR_ts_t0.1.png'
fd='runB_RMSE_ts_t0.1.png'
#fd='runB_MAE_ts_t0.1.png'
fe='runB_fractional_bias_ts_t0.1.png'
ff='runB_areafc_ts_t0.1.png'

cp ../$fa ./
cp ../$fb ./
cp ../$fcc ./
cp ../$fd ./
cp ../$fe ./
cp ../$ff ./

pntsz=20
pntsz2=30
lfh=+10
#lfh=-200
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
        fcc.png

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
        -trim fcc.png \
        -trim fd.png\
        -trim fe.png\
        -trim ff.png\
        -geometry 300x200 -tile 2x3 statsB.png

display statsB.png

