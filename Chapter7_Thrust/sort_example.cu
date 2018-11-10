/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc sort_example.cu -o sort_example
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

//*********************************************************
template < typename T> void print (char *s, thrust::host_vector < T > &v)
{
  cout << s ;
  thrust::copy (v.begin (), v.end (), ostream_iterator < T > (cout, " "));
  cout << endl;
}

//*********************************************************

int main ()
{
  int salary[] = { 1000, 2000, 1001, 2000, 3000, 5000 };
  int numItems = sizeof (salary) / sizeof (int);
  thrust::host_vector < int >h_salary (salary, salary + numItems);
  int SSN[] = { 212, 122, 34, 456, 890, 102 };
  thrust::host_vector < int >h_SSN (SSN, SSN + numItems);

  thrust::device_vector < int >d_salary (h_salary);
  thrust::device_vector < int >d_SSN (h_SSN);

  //-------------------------------
  // Example - thrust::sort_by_key
  thrust::sort_by_key (d_salary.begin (), d_salary.end (), d_SSN.begin ());
  h_salary = d_salary;
  h_SSN = d_SSN;
  print("Keys : ", h_salary);
  print("Values : ", h_SSN);
  //Output is:
  // Keys : 1000 1001 2000 2000 3000 5000 
  // Values : 212 34 122 456 890 102 

  //-------------------------------
  // Example - thrust::is_sorted
  cout << thrust::is_sorted (h_salary.begin (), h_salary.end ()) << endl;
  //Output is:
  // 1

  //-------------------------------  
  // Searching on the device : SCALAR VERSIONS
  thrust::device_vector < int >::iterator i = thrust::lower_bound (d_salary.begin (), d_salary.end (), 1500);
  cout << "Found at index " << i - d_salary.begin () << " Value " << *i << endl;
  //Output is:
  // Found at index 2 Value 2000

  i = thrust::upper_bound (d_salary.begin (), d_salary.end (), 2500);
  cout << "Found at index " << i - d_salary.begin () << " Value " << *i << endl;
  //Output is:
  // Found at index 4 Value 3000

  thrust::pair < thrust::device_vector < int >::iterator, thrust::device_vector < int >::iterator > p;
  p = thrust::equal_range (d_salary.begin (), d_salary.end (), 2000);
  cout << "Range equal to item is between indices " << p.first - d_salary.begin () << " " << p.second - d_salary.begin () << endl;
  //Output is:
  // Range equal to item is between indices 2 4

  p = thrust::equal_range (d_salary.begin (), d_salary.end (), 2222);
  cout << "Range equal to item is between indices " << p.first - d_salary.begin () << " " << p.second - d_salary.begin () << endl;
  //Output is:
  // Range equal to item is between indices 4 4
  
  cout << thrust::binary_search (d_salary.begin (), d_salary.end (), 1500) << endl;
  //Output is:
  // 0

  //-------------------------------
  // Searching on the host
  thrust::host_vector < int >::iterator j = thrust::lower_bound (h_salary.begin (), h_salary.end (), 2000);
  cout << j - h_salary.begin () << " " << *j << endl;
  //Output is:
  // 2 2000
   
  //-------------------------------  
  // Searching on the device : VECTOR VERSIONS
  thrust::device_vector<int> itemsToLook(10);
  thrust::sequence(itemsToLook.begin(), itemsToLook.end(), 0, 500);
  thrust::device_vector<int> results;
  thrust::host_vector<int> h_r;
  results.resize(itemsToLook.size());
  h_r = itemsToLook;
  print("Searching for : ", h_r);
  //Output is:
  // Searching for : 0 500 1000 1500 2000 2500 3000 3500 4000 4500 
  
  
  thrust::lower_bound (d_salary.begin (), d_salary.end (), itemsToLook.begin(), itemsToLook.end(), results.begin());
  h_r = results;
  print("Lower bounds : ", h_r);
  //Output is:
  // Lower bounds : 0 0 0 2 2 4 4 5 5 5 
  
  thrust::upper_bound (d_salary.begin (), d_salary.end (), itemsToLook.begin(), itemsToLook.end(), results.begin());
  h_r = results;
  print("Upper bounds : ", h_r);
  //Output is:
  // Upper bounds : 0 0 1 2 4 4 5 5 5 5 
  
  thrust::binary_search (d_salary.begin (), d_salary.end (), itemsToLook.begin(), itemsToLook.end(), results.begin());
  h_r = results;
  print("Binary search results : ", h_r);
  //Output is:
  // Binary search results : 0 0 1 0 1 0 1 0 0 00
     
  return 0;
}
