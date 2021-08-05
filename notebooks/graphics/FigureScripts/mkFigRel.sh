#!/bin/bash -f
##


fa='reliability_A_t2_3_4_5bc0p2.png'
fb='reliability_A_t6_7_8_9bc0p2.png'
fcc='reliability_A_t10_11_12_13bc0p2.png'
fd='reliability_number_A_t2_3_4_5bc0p2.png'
fe='reliability_number_A_t6_7_8_9bc0p2.png'
ff='reliability_number_A_t10_11_12_13bc0p2.png'

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
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(c)' \
        fb.png

convert $fcc -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(e)' \
        fc.png

convert $fd -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(b)' \
        fd.png

convert $fe -gravity $grav\
        -font Times-Roman -pointsize $pntsz -annotate $lfh+$cen '(d)' \
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
        -geometry 600x400 -tile 3x2 reliabilityA.png

display reliabilityA.png

#display stats.png
