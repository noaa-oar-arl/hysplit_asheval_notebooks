#!/bin/bash -f
##


fa='emissionsB_totalmass.png'
fcc='emissionsB_legend.png'
fb='emissionsB.png'

cp ../../$fa ./
cp ../../$fb ./
cp ../../$fcc ./


#--------------------------------------------------------------
#--------------------------------------------------------------


montage -trim $fcc \
        -trim $fa \
        -geometry 700x250 -tile 1x2 temp.png

montage -trim $fb \
        -trim temp.png \
        -geometry 700x700 -tile 2x1 emisB.png


display emisB.png

#montage -trim temp1.png \
#        -trim $fcc \
#        -geometry 400x400 -tile 2x1 emisM.png

#display emisM.png
#display stats.png
