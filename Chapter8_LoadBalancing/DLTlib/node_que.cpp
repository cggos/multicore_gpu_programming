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

#include <stdio.h>
#include <stdlib.h>

#define MAX_QUEUES 10
#define MAX_QUEUE_ELEM 200

#ifndef NULL
 #define NULL 0
#endif

struct Node;

// global data handled by Queue class
char init_queue=1;

int que_used[MAX_QUEUES]; // which queue is used
int que_head[MAX_QUEUES];
int que_tail[MAX_QUEUES];
int que_num_elem[MAX_QUEUES];
Node *que_elem[MAX_QUEUES][MAX_QUEUE_ELEM];

class Queue
{
 public:
  int my_queue;   // only one handle needed for queue operations

  Queue();
  ~Queue();
  void Insert(Node *);
  void Insert(Node &);
  Node * Pop();
  char IsEmpty();
};

//-------------------------------------------------------------
Queue::Queue()
 {
  int i;

   if(init_queue)
    {
      init_queue=0;
      for(i=1;i<MAX_QUEUES;i++) que_used[i]=0;
      que_used[0]=1;
      my_queue=0;
    }
   else
    {
      i=0;
      while((i<MAX_QUEUES) && que_used[i]) i++;
      if(i==MAX_QUEUES)
        {
          printf("Not enough queues\n");
          exit(1);
        }
      que_used[i]=1;
      my_queue=i;
    }
   que_head[my_queue]=0;
   que_tail[my_queue]=0;
   que_num_elem[my_queue]=0;
 }
//-------------------------------------------------------------
void Queue::Insert(Node &x)
 {
   que_elem[my_queue][que_tail[my_queue]]=&x;
   que_tail[my_queue] = (que_tail[my_queue]+1) % MAX_QUEUE_ELEM;
   if((++que_num_elem[my_queue])>MAX_QUEUE_ELEM)
     {
       printf("No queue space\n");
       exit(1);
     }
 }
//-------------------------------------------------------------
void Queue::Insert(Node *x)
 {
   que_elem[my_queue][que_tail[my_queue]]=x;
   que_tail[my_queue] = (que_tail[my_queue]+1) % MAX_QUEUE_ELEM;
   if((++que_num_elem[my_queue])>MAX_QUEUE_ELEM)
     {
       printf("No queue space\n");
       exit(1);
     }
 }
//-------------------------------------------------------------
Node * Queue::Pop()
 {
  Node *x;

  if(que_num_elem[my_queue])
   {
    que_num_elem[my_queue]--;
    x=que_elem[my_queue][que_head[my_queue]];
    que_head[my_queue] = (que_head[my_queue]+1) % MAX_QUEUE_ELEM;
   }
  else
   {
     printf("Attempted Pop from empty queue\n");
   }
  return(x);
 }
//-------------------------------------------------------------
char Queue::IsEmpty()
 {
   if(que_num_elem[my_queue]) return(0);
   return(1);
 }
//-------------------------------------------------------------
Queue::~Queue()
{
  que_used[my_queue]=0;
}
