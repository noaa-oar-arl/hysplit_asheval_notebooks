## Supplementary Material for 
Evaluation and Bias Correction of Probabilistic Volcanic Ash Forecasts
by Alice Crawford, Tianfeng Chai, Binyu Wang, Allison Ring1, Barbara Stunder, Christopher Loughner, Michael Pavolonis, and Justin Sieglaff

### Dependencies

* python modules
    * jupyter, datetime, matplotlib, seaborn
    * utilhysplit 
    * monetio
* data files with meteorology, satellite retrievals and HYSPLIT output can be obtained by contacting Alice Crawford.

* Imagemagick was used for creating many of the multi-panel plots

### Contents

#### notebooks directory
* jupyter notebooks
    * notebooks for inversion algorithm
        * BezyInverseRunB
        * BezyInverseRunC
        * BezyInverseRunD
        * BezyInverseRunE
        * BezyInverseRunPartSize
        * BezyInverseRunM
    * notebook for plotting meteorological data (Appendix A)
        * GEFS_bezy_comparison.ipynb
    * notebooks for particle separation (Appendix B)
        * particle_spearation.ipynb (Figure B1 and B2)
    * notebooks for VOLCAT data
        * Bezy.ipynb 
            * Figure 1
            * Figure 2
    * notebooks for evaluation
        * BezyEvalRunMfinal.ipynb
            * Figure 11
        * BezyEvalRunAfina.ipynb
            * Figure 6
            * Figure 12
        * 
    
    * graphics/FigureScripts directory
        * bash scripts for creating multi-panel plots with ImageMagick.
     

#### Directories with files for running HYSPLIT
* RunA
* RunM

## Disclaimer
"This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government."

