#!/bin/bash -f
##


fa='emissionsM_totalmass.png'
fcc='emissionsM_legend.png'
fb='emissionsM.png'

cp ../$fa ./
cp ../$fb ./
cp ../$fcc ./


#--------------------------------------------------------------
#--------------------------------------------------------------


montage -trim $fb \
        -trim $fa \
        -geometry 700x250 -tile 1x2 emisM.png

display emisM.png

#montage -trim temp1.png \
#        -trim $fcc \
#        -geometry 400x400 -tile 2x1 emisM.png

#display emisM.png
#display stats.png
