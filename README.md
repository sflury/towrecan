# towrecen
The towrecen code characterizes the scatter in data by quantifying the dispersion with respect to some fiducial value. There are two possible approaches which `towrecen` implements:
+ determine the (signed) minimal orthogonal distance between each datum and some reference function
+ determine the orthogonal distance which maximizes the variance with respect to the mean or median of the data
The former involves root-finding to determine _where_ a function gets minimized. This approach is inherently model-dependent. The latter lends itself naturally to principal component analysis (PCA -- perhaps a foregone conclusion for the stats-oriented folks). This approach is model-independent. There are pros and cons to each approach, and it is up to the user to determine which method is more insightful/meaningful given their data.

In the case of the explicity orthogonal distance method, if no model is provided, `towrecen` will produce a model by performing a simple matrix-algebra linear least squares fit. PCA can also optionally take a model instead of using its non-parametric approach of mean subtraction; however, this is not recommended.

## Example Usage
``` python
from numpy.random import rand, randn, seed
from towrecen import *
seed(123)
import sys,os
# function for orthogonal distance
# here using Nicholls+ 2017
f = lambda x: log10(10**-1.732+10**((x-12)+2.19))
# generate fake data
x  = 1.5*rand(50)+7.5
y  = f(x) + randn(len(x))*0.25
xe = 0.05*rand(len(x))+0.05
ye = 0.05*rand(len(x))+0.05
# instantiate towrecen
disp = towrecen(x,y,fun=f)
# plot data to illustrate scatter
disp.plotDistCompare()
```
<img width="583" alt="image" src="https://github.com/sflury/towrecen/assets/42982705/14677bf2-8064-4984-aee0-defbc1254887">

## Whence the Name `towrecen`
I studied Old, Middle, and Early Modern English in college, both from a linguistic angle and from a literary one. I wanted to reconnect to those scholastic roots. The name `towrecen` is an Old English word meaning 'to scatter' or 'to disperse'. The word contains connotations of sowing fields. I often picture data as having been scattered in such a way, with the Universe casting galactic seeds to be nurtured through gas accretion and the subtle laws of physics. Ours is to determine whether the dispersion is physical (like the mass-metallicity relation) and if so, why.

## Citing `towrecen`
This code has its origins in my Master's thesis; however, it first appeared in 

``` bibtex
@ARTICLE{AuthorYear,
       author = {{Flury}, Sophia R. and ??},
        title = "{}",
      journal = {},
         year = 202X,
        month = ,
       volume = {},
       number = {},
        pages = {},
          doi = {} }
```

## Licensing
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
