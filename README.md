# towrecan
The towrecan code characterizes the scatter in data by quantifying the dispersion with respect to some fiducial value. There are two possible approaches which `towrecan` implements:
+ determine the (signed) minimal orthogonal distance between each datum and some reference function
+ determine the orthogonal distance which maximizes the variance with respect to the mean or median of the data

The former involves root-finding to determine _where_ a function gets minimized. This approach is inherently model-dependent. The latter lends itself naturally to principal component analysis (PCA -- perhaps a foregone conclusion for the stats-oriented folks). This approach is model-independent. There are pros and cons to each approach, and it is up to the user to determine which method is more insightful/meaningful given their data and objectives.

In the case of the explicity orthogonal distance method, if no model is provided, `towrecan` will produce a model by performing a Demming regression fit, which assumes a simle first-order polynomial and accounts for scatter and uncertainty in both the abscissa (independent) and ordinate (dependent) variables. PCA implicitly contains a similar "fit" when reprojecting the data via spectral decomposition. The best-"fit" coefficients are contained in the eigenvectors of the covariance matrix of the dependent and independent variables. This PCA "fit" is equivalent to total least squares and is often similar to (but not the same as) Demming regression.

## Example Usage
``` python
from numpy.random import rand, randn, seed
from towrecan import *
seed(123)
# function for orthogonal distance
# here using Nicholls+ 2017
f = lambda x: log10(10**-1.732+10**((x-12)+2.19))
# generate fake data
x  = 1.5*rand(50)+7.5
y  = f(x) + randn(len(x))*0.25
xe = 0.05*rand(len(x))+0.05
ye = 0.05*rand(len(x))+0.05
# instantiate towrecan
disp = towrecan(x,y,fun=f,xerr=xe,yerr=ye)
# plot data to illustrate scatter and
# minimized Euclidean distances
disp.plotDistOrtho()
```
<img width="400" alt="image" src="https://github.com/sflury/towrecan/assets/42982705/aeb72b15-0956-48ed-8fc8-83e258bf23f0">

``` python
# compare dispersion measurements via Euclidean distance minimization and PCA
disp.plotDistCompare()
```
<img width="400" alt="image" src="https://github.com/sflury/towrecan/assets/42982705/14677bf2-8064-4984-aee0-defbc1254887">

## Whence the Name `towrecan`
I studied Old, Middle, and Early Modern English in college, both from a linguistic angle and from a literary one. I wanted to reconnect to those scholastic roots. The name `towrecan` is an Old English word meaning 'to scatter', 'to disperse', or 'to drive asunder'. The word contains connotations of a driving force (think of the word 'wreck', which has its origins in 'towrecan'). I picture extreme phenomena like supernovae or active galactic nuclei actively causing unique observable conditions through brute force, wrecking the status quo of their galaxies and shaping the scaling relations we observe today. Ultimately ours is to determine whether the dispersion we see in our data is physical (like the mass-metallicity relation) and if so, why.

## Citing `towrecan`
The `towrecan` code has its origins in my Master's thesis; however, it will first appear "publicly" in the near future. As of now, please cite this repository when utilizing `towrecan`. Future work will present details of this code with application to an astrophysical context.

``` bibtex
@SOFTWARE{Flury2025,
       author = {{Flury}, Sophia R.?},
        title = "{towrecan: Quantifying Dispersion in Data}",
         year = 2025,
        month = march,
      version = {1.0.1},
          url = {https://github.com/sflury/towrecan},
          doi = {10.5281/zenodo.15057953} }
```

Zenodo badge:
[![DOI](https://zenodo.org/badge/767437829.svg)](https://doi.org/10.5281/zenodo.15057953)

## Licensing
<a href="https://github.com/sflury/towrecan">towrecan</a> © 2025 by <a href="https://sflury.github.io">Sophia Flury</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International</a>

<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nd.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;">

This license enables reusers to copy and distribute towrecan in any medium or format in unadapted form only, for noncommercial purposes only, and only so long as attribution is given to the creator. CC BY-NC-ND 4.0 includes the following elements:

BY: credit must be given to the creator.
NC: Only noncommercial uses of the work are permitted.
ND: No derivatives or adaptations of the work are permitted.

You should have received a copy of the CC BY-NC-ND 4.0 along with this program. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
