/*
    Part of the DLTlib library
    Copyright (C) 2014, Gerassimos Barlas
    Contact : gerassimos.barlas@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses
*/

/*-- Random number generation routines taken from Numerical Recipes in C --*/
#ifndef CUSTOM_RANDOM

#ifdef __cplusplus
extern "C" {
#endif

#define CUSTOM_RANDOM

    float ran2(long *idum);
    float gasdev(long *idum);
    float gauss();

    double find_z(double cumulative);

#ifdef __cplusplus
}
#endif

#endif

