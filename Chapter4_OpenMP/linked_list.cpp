/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp linked_list.cpp -o linked_list
 ============================================================================
 */
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

// template structure for a list's node
template <class T>
struct Node
{
  T info;
  Node *next;
};
//---------------------------------------
// Appends a value at the end of a list pointed by the head *h
template <class T>
void append (int v, Node<T> ** h)
{
  Node<T> *tmp = new Node<T> ();
  tmp->info = v;
  tmp->next = NULL;

  Node<T> *aux = *h;
  if (aux == NULL)              // first node in list
    *h = tmp;
  else
    {
      while (aux->next != NULL)
        aux = aux->next;
      aux->next = tmp;
    }
}

//---------------------------------------
// function stub for processing a node's data
template <class T>
void process (Node<T> * p)
{
#pragma omp critical
  cout << p->info << " by thread " << omp_get_thread_num () << endl;
}

//---------------------------------------
int main (int argc, char *argv[])
{
  // build a sample list
  Node<int> *head = NULL;
  append (1, &head);
  append (2, &head);
  append (3, &head);
  append (4, &head);
  append (5, &head);

#pragma omp parallel
  {
#pragma omp single
    {
      Node<int> *tmp = head;
      while (tmp != NULL)
        {
#pragma omp task
          process (tmp);
          tmp = tmp->next;
        }

    }
  }

  return 0;
}
