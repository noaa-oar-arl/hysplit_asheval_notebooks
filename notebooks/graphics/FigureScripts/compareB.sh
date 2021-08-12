#!/bin/bash -f
##


fa='pcolormesh_runB_20201022_00_RunB_2_3_4_TFw5.png'
fb='pcolormesh_runB_20201022_04_RunB_2_3_4_TFw5.png'
fcc='pcolormesh_runB_20201022_06_RunB_2_3_4_TFw5.png'
fd='pcolormesh_runB_20201022_08_RunB_2_3_4_TFw5.png'
fe='pcolormesh_runB_20201022_10_RunB_2_3_4_TFw5.png'
ff='pcolormesh_runB_20201022_12_RunB_2_3_4_TFw5.png'

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
        -geometry 600x200 -tile 1x6 compareB.png

display compareB.png

