# towrecen
The towrecen code characterizes the scatter in data by quantifying the dispersion with respect to some fiducial value. There are two possible approaches which `towrecen` implements:
+ determine the (signed) minimal orthogonal distance between each datum and some reference function
+ determine the orthogonal distance which maximizes the variance with respect to the mean or median of the data
The former involves root-finding to determine _where_ a function gets minimized. The latter lends itself naturally to principal component analysis (PCA -- perhaps a foregone conclusion for the stats-oriented folks).

## Example Usage
``` python
from numpy.random import rand, randn, seed
from towrecen import *
seed(123)
import sys,os
# Nicholls+ 2017 N/O vs O/H scaling relation
f = lambda x: log10(10**-1.732+10**((x-12)+2.19))
# generate fake data
x  = 1.5*rand(5)+7.5
y  = 1.5*rand(5)-1.75
# instantiate towrecen
disp = towrecen(x,y,fun=f)
# plot data to illustrate scatter
disp.pplot()
```
<img width="412" alt="image" src="https://github.com/sflury/towrecen/assets/42982705/13888594-265c-40be-a390-1b2a22c8be3b">

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
