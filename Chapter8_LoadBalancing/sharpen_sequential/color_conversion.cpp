#include <math.h>

void RGB2XYZ(int inp_R, int inp_G, int inp_B, float &out_X, float &out_Y, float &out_Z)  
{ 
float var_R, var_G, var_B;
var_R = ( inp_R / 255.0 );        //inp_R = From 0 to 255
var_G = ( inp_G / 255.0 );        //inp_G = From 0 to 255
var_B = ( inp_B / 255.0 );        //inp_B = From 0 to 255

if (var_R > 0.03928) var_R = pow( ( var_R + 0.055 ) / 1.055 ,  2.4);
else                 var_R = var_R / 12.92;
if (var_G > 0.03928) var_G = pow( ( var_G + 0.055 ) / 1.055 ,  2.4);
else                 var_G = var_G / 12.92;
if (var_B > 0.03928) var_B = pow( ( var_B + 0.055 ) / 1.055 , 2.4);
else                 var_B = var_B / 12.92;

var_R = var_R * 100;
var_G = var_G * 100;
var_B = var_B * 100;

//Observer. = 2°, Illuminant = D65
out_X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
out_Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
out_Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;
}
//**************************************************************************
void XYZ2RGB(float inp_X, float inp_Y, float inp_Z, int &out_R, int &out_G, int &out_B)  
{ 
float var_X, var_Y, var_Z;
float var_R, var_G, var_B;

var_X = inp_X / 100;        //inp_X = From 0 to ref_X
var_Y = inp_Y / 100;        //inp_Y = From 0 to ref_Y
var_Z = inp_Z / 100;        //inp_Z = From 0 to ref_Y

var_R = var_X *  3.2410 + var_Y * -1.5374 + var_Z * -0.4986;
var_G = var_X * -0.9692 + var_Y *  1.8760 + var_Z *  0.0416;
var_B = var_X *  0.0556 + var_Y * -0.2040 + var_Z *  1.0570;

if( var_R > 0.00304 ) var_R = 1.055 * pow( var_R , ( 1 / 2.4 ) ) - 0.055;
else                  var_R = 12.92 * var_R;
if( var_G > 0.00304 ) var_G = 1.055 * pow( var_G , ( 1 / 2.4 ) ) - 0.055;
else                  var_G = 12.92 * var_G;
if( var_B > 0.00304 ) var_B = 1.055 * pow( var_B , ( 1 / 2.4 ) ) - 0.055;
else                  var_B = 12.92 * var_B;

out_R = int(var_R * 255);
out_G = int(var_G * 255);
out_B = int(var_B * 255);
}
//**************************************************************************
void XYZ2LAB(float inp_X, float inp_Y, float inp_Z, float &out_L, float &out_a, float &out_b)  
{ 
float ref_X, ref_Y, ref_Z;
float var_X, var_Y, var_Z;

ref_X =  95.047;       //Observer. = 2°, Illuminant = D65
ref_Y = 100.000;
ref_Z = 108.883;

var_X = inp_X / ref_X;        //inp_X = From 0 to ref_X
var_Y = inp_Y / ref_Y;        //inp_Y = From 0 to ref_Y
var_Z = inp_Z / ref_Z;        //inp_Z = From 0 to ref_Z

if ( var_X > 0.008856 ) var_X = pow(var_X, 1/3.0 );
else                    var_X = ( 7.787 * var_X ) + ( 16 / 116.0 );
if ( var_Y > 0.008856 ) var_Y = pow(var_Y, 1/3.0 );
else                    var_Y = ( 7.787 * var_Y ) + ( 16 / 116.0 );
if ( var_Z > 0.008856 ) var_Z = pow(var_Z, 1/3.0 );
else                    var_Z = ( 7.787 * var_Z ) + ( 16 / 116.0 );

out_L = ( 116 * var_Y ) - 16;
out_a = 500 * ( var_X - var_Y );
out_b = 200 * ( var_Y - var_Z );
}
//**************************************************************************
void LAB2XYZ(float inp_L, float inp_a, float inp_b, float &out_X, float &out_Y, float &out_Z)  
{ 
float ref_X, ref_Y, ref_Z;
float var_X, var_Y, var_Z;

ref_X =  95.047;       //Observer. = 2°, Illuminant = D65
ref_Y = 100.000;
ref_Z = 108.883;

//inp_L = CIE-L*
//inp_a = CIE-a*
//inp_b = CIE-b*

var_Y = ( inp_L + 16 ) / 116;
var_X = inp_a / 500 + var_Y;
var_Z = var_Y - inp_b / 200;

if ( pow(var_Y, 3 ) > 0.008856 ) var_Y = pow(var_Y, 3);
else                          var_Y = ( var_Y - 16 / 116.0 ) / 7.787;
if ( pow(var_X, 3 ) > 0.008856 ) var_X = pow(var_X ,3);
else                          var_X = ( var_X - 16 / 116.0 ) / 7.787;
if ( pow(var_Z, 3 ) > 0.008856 ) var_Z = pow(var_Z, 3);
else                          var_Z = ( var_Z - 16 / 116.0 ) / 7.787;

out_X = ref_X * var_X;
out_Y = ref_Y * var_Y;
out_Z = ref_Z * var_Z;
}
//**************************************************************************
void LAB2RGB(float inp_L, float inp_a, float inp_b, int &out_R, int &out_G, int &out_B)  
{
   float X, Y, Z;
   LAB2XYZ(inp_L, inp_a, inp_b, X, Y, Z);
   XYZ2RGB(X,Y,Z,out_R, out_G, out_B);
}
//**************************************************************************
void RGB2LAB(int inp_R, int inp_G, int inp_B, float &out_L, float &out_a, float &out_b)  
{ 
   float X, Y, Z;
   RGB2XYZ(inp_R, inp_G, inp_B, X, Y, Z);
   XYZ2LAB(X,Y,Z,out_L, out_a, out_b);
}
