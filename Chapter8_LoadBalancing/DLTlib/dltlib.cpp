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

#include "node_que.cpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include "random.c"
#include "dltlib.h"

extern "C" {
#include <glpk.h>
}

#include <vector>
#include <algorithm>

using namespace std;

extern long global_random_seed;

int sched_lib_max_node_degree = MAX_NODE_DEGREE; // this controls the maximum
// degree of a node
// used for avoiding excessive mem allocs
Node *nodes_array[MAX_NUM_NODES];
Node *leaf_array[MAX_NUM_NODES];

Node *is_cond[MAX_NUM_NODES];
char cond_pattern[MAX_NUM_NODES + 1];
char cond_best[MAX_NUM_NODES];
char usage_best[MAX_NUM_NODES];
char usage_examined[MAX_NUM_NODES + 1];
double __prob_gain; // local possible gain for one node
//**************************************************************

Network& Network::operator =(Network& x) {
    Queue temp_queue;
    Node *temp;
    int i;

    // eliminate all previous info
    head = NULL;
    tail = NULL;
    redundant = NULL; // redundant nodes are not copied

    for (i = 0; i <= MAX_NUM_NODES; i++) node_usage[i] = 0;

    temp_queue.Insert(x.head);
    while (!temp_queue.IsEmpty()) {
        temp = temp_queue.Pop(); // pops a node ready for aggregate calculations
        InsertNode(temp->name, temp->power, temp->e0, temp->parent->name, temp->l2par, temp->fe, temp->through);
        if (temp->through != 2) // corrects a side effect of InsertNode 
            tail->through = temp->through;
        tail->part = temp->part; // used by Quantify
        tail->L = temp->L;
        tail->Lint = temp->Lint;
        for (i = 0; i < temp->degree; i++)
            temp_queue.Insert(temp->child[i]);
    }

    return (*this);
}
//---------------------------------------------
// Checks a solution given. Returns 1 if ok

bool Network::CheckSolution() {
    Node *temp = head;
    double calc_L = 0;
    double aux;
    char alarm = 0;

    while (temp && !alarm) {
        if (!temp->through) {
            if (temp->degree) aux = temp->part * temp->L;
            else aux = temp->part * temp->parent->L;
            if (aux < 0) alarm = 1;
            calc_L += aux;
        }
        temp = temp->next_n;
    }
    if ((fabs(head->L - calc_L) < 2) && !alarm) return (1);
    else return (0);
}
//---------------------------------------------

Network::Network() {
    int i;
    head = NULL;
    tail = NULL;
    redundant = NULL;
    clipping = 1;
    for (i = 0; i <= MAX_NUM_NODES; i++) node_usage[i] = 0;
}
//---------------------------------------------

void Network::AuxInsertNode(char *c, double speed, double e, double link2parent) {
    Node *temp;
    int i;

    i = strlen(node_usage);
    temp = &netnode[i];
    node_usage[i] = 1;

    strncpy(temp->name, c, MAX_NAME);
    temp->power = speed;
    temp->next_n = NULL;
    temp->e0 = e;
    temp->l2par = link2parent;
    temp->degree = 0;
    temp->visited = 0;
    if (head) {
        tail->next_n = temp;
        temp->ID = tail->ID + 1;
        tail = temp;
    } else {
        head = tail = temp;
        temp->ID = 0;
    }
}
//**************************************************************

Node* Network::InsertNode(char *c, double speed, double e, char *parent_name, double link2parent, bool front_end, bool thru) {
    Node *temp;

    (*this).AuxInsertNode(c, speed, e, link2parent);
    temp = head;
    if (parent_name)
        while (temp && strncmp(temp->name, parent_name, MAX_NAME)) temp = temp->next_n;
    else
        temp = NULL;
    if (temp) {
        temp->child[temp->degree] = tail;
        temp->link[temp->degree++] = link2parent;
    }
    tail->parent = temp;
    tail->fe = front_end;
    if (thru) tail->through = 2; // forced pass through node
    else tail->through = 0;
    return (tail);
}
//---------------------------------------------

Node* Network::InsertNode(char *c, double speed, double e, Node *parent, double link2parent, bool front_end, bool thru) {
    (*this).AuxInsertNode(c, speed, e, link2parent);
    parent->child[parent->degree] = tail;
    parent->link[parent->degree++] = link2parent;
    tail->parent = parent;
    if (thru) tail->through = 2; // forced pass through node
    else tail->through = 0;
    tail->fe = front_end;
    return (tail);
}
//---------------------------------------------
// Prints the solution acquired. If parameter == 0 then the load is arbitrary divisible

void Network::PrintSolution(bool quantized) {
    Node *temp;
    double aux;

    if (quantized) {
        temp = head;
        printf("Load distribution for %li units\n", temp->Lint);
        while (temp) {
            if (temp->degree) // internal node
                aux = temp->part * temp->Lint;
            else
                aux = temp->part * temp->parent->Lint;
            if (aux > 0.1)
                printf("Processor %s gets %.0f\n", temp->name, aux);
            temp = temp->next_n;
        }
    } else {
        double Lsum = 0;
        if ((head != tail) || head->next_n) {
            temp = head;
            printf("Running time %f\n", temp->aggregate_p * (temp->L + temp->aggregate_e));
            while (temp) {
                if (temp->degree) // internal node
                    aux = temp->part * temp->L;
                else
                    aux = temp->part * temp->parent->L;
                printf("Processor %s with %lf gets %f%% for a total of %lf load units\n", temp->name, temp->e0, temp->part, aux + temp->e0);
                Lsum += aux + temp->e0;
                temp = temp->next_n;
            }
            printf("Total load %lf\n", Lsum);
        } else
            printf("Running time %f on HOST\n", head->power * (head->L + head->e0));
    }
}
//---------------------------------------------
// sort the children of each internal node so that optimal
// sequencing can occur for query processing
// Sorting must occur after aggregate fields of leaf node are set

void QuerySort(Node *temp) {
    Node *temp2;
    int i, j;
    double aux, min_lp;
    int mark;

    if (temp->degree > 1) {
        // bubblesort
        for (i = 0; i < temp->degree - 1; i++) {
            mark = i;
            min_lp = temp->link[i] * temp->child[i]->aggregate_p;
            for (j = i + 1; j < temp->degree; j++) {
                aux = temp->link[j] * temp->child[j]->aggregate_p;
                if (min_lp > aux) {
                    min_lp = aux;
                    mark = j;
                }
            }
            if (mark != i) // switch pointers
            {
                temp2 = temp->child[i];
                aux = temp->link[i];
                temp->child[i] = temp->child[mark];
                temp->link[i] = temp->link[mark];
                temp->child[mark] = temp2;
                temp->link[mark] = aux;
            }
        }
    }
}
//---------------------------------------------
// Returns 1 if the condition a*L*l+e1*p1-e2*p2>0 holds
// for node temp. Then T4 assignment is better

bool CheckT4(Node *temp, double a) {
    int i, j;
    double l, e1, e2, p1, p2, part1, part2;
    double L; // the load assigned to subtree rooted at temp
    double aux, min_pe, max_pe;
    char flag = 0;

    __prob_gain = 0;
    if (temp->degree > 1) {
        l = temp->link[0];
        L = temp->L;
        for (i = 0; i < temp->degree - 1; i++) {
            e1 = temp->child[i]->aggregate_e;
            e2 = temp->child[i + 1]->aggregate_e;
            p1 = temp->child[i]->aggregate_p;
            p2 = temp->child[i + 1]->aggregate_p;
            if (temp->child[i]->degree) part1 = temp->child[i]->L;
            else part1 = temp->child[i]->part *L;
            if (temp->child[i + 1]->degree) part2 = temp->child[i + 1]->L;
            else part2 = temp->child[i + 1]->part *L;

            if (a * (part1 + part2) * l + e1 * p1 - e2 * p2 < 0) {
                flag = 1;
                i = temp->degree;
            }
        }
    }
    if (flag) {
        for (i = 0; i < temp->degree; i++)
            for (j = 0; j < temp->degree; j++) {
                if (i != j) {
                    e1 = temp->child[i]->aggregate_e;
                    e2 = temp->child[j]->aggregate_e;
                    p1 = temp->child[i]->aggregate_p;
                    p2 = temp->child[j]->aggregate_p;
                    if (temp->child[i]->degree) part1 = temp->child[i]->L;
                    else part1 = temp->child[i]->part *L;
                    if (temp->child[j]->degree) part2 = temp->child[j]->L;
                    else part2 = temp->child[j]->part *L;

                    aux = (a * (part1 + part2) * l + e1 * p1 - e2 * p2) * l / ((a + 1) * l + p1 + p2);
                    __prob_gain = MIN(__prob_gain, aux);
                }
            }
        return (0);
    } else return (1);
}
//---------------------------------------------
// sort the children of each internal node so that optimal
// sequencing can occur for image processing
// Sorting must occur after aggregate fields of leaf node are set

void ImageSort(Node *temp) {
    Node *temp2;
    int i, j;
    double aux, min_ep;
    int mark;

    if (temp->degree > 1) {
        // bubblesort
        for (i = 0; i < temp->degree - 1; i++) {
            mark = i;
            min_ep = temp->child[i]->aggregate_e * temp->child[i]->aggregate_p;
            for (j = i + 1; j < temp->degree; j++) {
                aux = temp->child[j]->aggregate_e * temp->child[j]->aggregate_p;
                if (min_ep > aux) {
                    min_ep = aux;
                    mark = j;
                }
            }
            if (mark != i) // switch pointers
            {
                temp2 = temp->child[i];
                aux = temp->link[i];
                temp->child[i] = temp->child[mark];
                temp->link[i] = temp->link[mark];
                temp->child[mark] = temp2;
                temp->link[mark] = aux;
            }
        }
    }
}
//---------------------------------------------
// If the plain flag is set no check for the through flag is made
// in order to speed up enumeration of the Optimum routine

void Network::SolveQuery(long L, double b, double d, bool plain) {
    Queue temp_queue;
    Node *temp;
    int i, j;
    int last_switched_node;

    temp = head;
    while (temp) {
        temp->visited = temp->degree;
        if (!plain)
            if (temp->through != 2) temp->through = 0;
        temp = temp->next_n;
    }

    temp = head;
    while (temp) {
        if (temp->degree == 0) {
            temp->aggregate_p = temp->power;
            temp->aggregate_e = temp->e0;
            temp->parent->visited--;
            if (!temp->parent->visited)
                temp_queue.Insert(temp->parent);
        }
        temp = temp->next_n;
    }

    // Solve the problem. N is temp->degree
    // The first while calculate the aggregate_p and aggregate_e
    while (!temp_queue.IsEmpty()) {
        temp = temp_queue.Pop(); // pops a node ready for aggregate calculations
        QueryAggregate(temp, b, d);
        if (temp->parent) {
            temp->parent->visited--;
            if (!temp->parent->visited)
                temp_queue.Insert(temp->parent);
        }
    }
    // In the second run the actual load each processor gets are calculated
    //temp contains now the root processor which is the last to pop
    // The loop is repeated if recalculation of aggregate values is necessary
    if (!plain) {
        last_switched_node = -1;
        do {
            valid = 1;
            temp = head;
            temp->L = L;
            temp_queue.Insert(temp);
            while (!temp_queue.IsEmpty() && (valid == 1)) // exits if cannot distribute
            {
                temp = temp_queue.Pop();
                if (temp->parent) temp->L = temp->part * temp->parent->L;
                valid = QueryPart(temp, b, d);

                if (valid == 2) {
                    if ((temp->through == 1) && (last_switched_node == temp->ID)) {
                        valid = 1;
                        last_switched_node = -1;
                    } else if (temp->through == 0) temp->through = 1;
                    else temp->through = 0;
                }

                // each child node will be examined next if it is not a leaf
                for (i = temp->degree - 1; i >= 0; i--)
                    if (temp->child[i]->degree) temp_queue.Insert(temp->child[i]);
            }
            if (valid == 2) // aggregate values recalculation
            {
                last_switched_node = temp->ID;
                while (temp) // recalculate up to the top node
                {
                    QueryAggregate(temp, b, d);
                    temp = temp->parent;
                }
                // Empty queue before next iteration
                while (!temp_queue.IsEmpty()) temp_queue.Pop();
            }
        } while (valid == 2);
    } else {
        valid = 1;
        temp = head;
        temp->L = L;
        temp_queue.Insert(temp);
        while (!temp_queue.IsEmpty() && (valid == 1)) // exits if cannot distribute
        {
            temp = temp_queue.Pop();
            if (temp->parent) temp->L = temp->part * temp->parent->L;
            valid = QueryPart(temp, b, d);

            // each child node will be examined next if it is not a leaf
            for (i = temp->degree - 1; i >= 0; i--)
                if (temp->child[i]->degree) temp_queue.Insert(temp->child[i]);
        }
        if (valid == 2) valid = 0;
    }
    // Empty queue in case error occured
    while (!temp_queue.IsEmpty()) temp_queue.Pop();
}
//---------------------------------------------

void Network::QueryAggregate(Node *temp, double b, double d) {
    Node *temp2;
    double s1_p;
    double s_l;
    double s_e;
    double s1_p_l;
    int i, j;

    QuerySort(temp);
    s1_p = s_e = 0;
    for (i = 0; i < temp->degree; i++) // i ranges 1..N
    {
        s_e += temp->child[i]->aggregate_e;
        s1_p += 1 / temp->child[i]->aggregate_p;
    }
    s_l = s1_p_l = 0;

    if (temp->through) {
        s_l = temp->link[0];
        for (i = 1; i < temp->degree; i++) // i ranges 2..N
        {
            s_l += temp->link[i];
            s1_p_l += s_l / temp->child[i]->aggregate_p;
        }
        temp->aggregate_p = 1 / s1_p;
        temp->aggregate_e = (b + d) * s1_p_l + s_e + (b + d) *
                temp->link[0] / temp->child[0]->aggregate_p;
    } else {
        // host is participating
        for (i = 0; i < temp->degree; i++) // i ranges 1..N
        {
            s_l += temp->link[i];
            s1_p_l += s_l / temp->child[i]->aggregate_p;
        }
        s_e += temp->e0;
        temp->aggregate_p = 1 / (s1_p + 1 / temp->power);
        if (temp->fe) // decide correct aggregation method
            temp->aggregate_e = (b + d) * s1_p_l + s_e;
        else
            temp->aggregate_e = (b + d)*(s1_p_l + s_l / temp->power) + s_e;
    }
}
//---------------------------------------------
// return value 0: invalid net no solution
//              1: ok
//              2: temp node set/reset to 'through' mode.
//'2' response requires recalculation of aggregate values from here to host node

int Network::QueryPart(Node *temp, double b, double d) {
    Node *temp1, *temp2;
    double s1_p;
    double s_l;
    double s_e;
    double s1_p_l;
    int i, j;
    double aux1, aux2, aux3, p1, e1;
    double min_part[MAX_NODE_DEGREE];
    int res = 1;

    // check possibility of solution
    if (temp->degree != 1) {
        for (i = 1; i < temp->degree; i++) {
            temp1 = temp->child[i - 1];
            temp2 = temp->child[i];
            aux1 = temp1->aggregate_p * temp1->aggregate_e - temp2->aggregate_p * temp2->aggregate_e - (b + d) * temp->link[i];
            aux2 = temp2->aggregate_p * temp->L;
            min_part[i] = MAX(0, aux1 / aux2);
        }
        temp1 = temp->child[0];
        aux1 = temp->power * temp->e0 - temp1->aggregate_p * temp1->aggregate_e;
        if (temp->fe)
            min_part[0] = MAX(0, (aux1 - b * temp->link[0]) / (temp->L * temp1->aggregate_p));
        else {
            aux2 = 0;
            for (i = 1; i < temp->degree; i++) aux2 += temp->link[i];
            min_part[0] = MAX(0, (aux1 + b * aux2) / (temp->L * temp1->aggregate_p));
        }
        aux1 = 0;
        for (i = 0; i < temp->degree; i++) aux1 += min_part[i];
        if (aux1 > 1) return (0);
    }


    // now solve. Solution must be found in any case just to verify that the through state is correct
    s_l = s1_p_l = 0;
    if (temp->fe) // decide correct method for calculation of parts
    {
        for (i = 0; i < temp->degree; i++) // i ranges 1..N
        {
            s_l += temp->link[i];
            s1_p_l += s_l / temp->child[i]->aggregate_p;
        }

        s1_p = s_e = 0;
        for (i = 0; i < temp->degree; i++) // i ranges 1..N
        {
            s_e += temp->child[i]->aggregate_e;
            s1_p += 1 / temp->child[i]->aggregate_p;
        }

        // this is part0 :
        temp->part = (temp->L + (b + d) * s1_p_l +
                s_e - temp->e0 * temp->power * s1_p)
                *(temp->aggregate_p) / (temp->power * temp->L);
        // now calculate for each child parti
        s_l = 0;
        for (i = 0; i <= temp->degree - 1; i++) // i ranges 1..N
        {
            s_l += temp->link[i];
            temp2 = temp->child[i];
            temp2->part = temp->power * temp->part -
                    (b + d) * s_l / temp->L +
                    (temp->e0 * temp->power - temp2->aggregate_e * temp2->aggregate_p) / temp->L;
            temp2->part /= temp2->aggregate_p;
        }
    } else {
        // calculates part0
        for (i = temp->degree - 1; i > 0; i--) // i ranges 2..N
        {
            s_l += temp->link[i];
            s1_p_l += s_l / temp->child[i - 1]->aggregate_p;
        }
        s_l += temp->link[0];

        s1_p = s_e = 0;
        for (i = temp->degree - 1; i >= 0; i--) // i ranges 1..N
        {
            s_e += temp->child[i]->aggregate_e;
            s1_p += 1 / temp->child[i]->aggregate_p;
        }

        temp->part = (temp->L - (b + d) * s1_p_l -
                temp->e0 * temp->power * s1_p + s_e)
                *(temp->aggregate_p) / (temp->L * temp->power);
        // now calculate for each child parti
        s_l = 0;
        for (i = temp->degree - 1; i >= 0; i--) // i ranges 1..N
        {
            if (i < temp->degree - 1)
                s_l += temp->link[i + 1];
            temp2 = temp->child[i];
            temp2->part = s_l * (b + d) / temp->L +
                    (temp->e0 * temp->power - temp2->aggregate_e * temp2->aggregate_p) / temp->L +
                    temp->power * temp->part;
            temp2->part /= temp2->aggregate_p;
        }
    }


    // check validity of solution
    for (i = 0; i < temp->degree; i++)
        if (temp->child[i]->part < 0) return (0);
    if (((temp->part < 0) && (temp->through == 0)) || // state switching possibly appropriate
            ((temp->part > 0) && (temp->through == 1)))
        res = 2;


    // solve again for inactive parent
    s_l = s1_p_l = 0;
    if (temp->through) {
        for (i = 1; i < temp->degree; i++) // i ranges 2..N
        {
            s_l += temp->link[i];
            s1_p_l += s_l / temp->child[i]->aggregate_p;
        }

        s1_p = s_e = 0;
        for (i = 1; i < temp->degree; i++) // i ranges 2..N
        {
            s_e += temp->child[i]->aggregate_e;
            s1_p += 1 / temp->child[i]->aggregate_p;
        }

        // this is part1 :
        temp->part = 0;
        temp1 = temp->child[0];
        p1 = temp1->aggregate_p;
        e1 = temp1->aggregate_e;
        temp1->part = temp->L + (b + d) * s1_p_l + s_e - s1_p * p1*e1;
        temp1->part *= 1 / (s1_p * p1 + 1) / temp->L;
        // now calculate for each child parti
        s_l = 0;
        for (i = 1; i <= temp->degree - 1; i++) // i ranges 2..N
        {
            s_l += temp->link[i];
            temp2 = temp->child[i];
            temp2->part = (p1 * temp1->part * temp->L - (b + d) * s_l +
                    (p1 * e1 - temp2->aggregate_e * temp2->aggregate_p)) /
                    (temp->L * temp2->aggregate_p);
        }

        // check validity of solution
        for (i = 0; i < temp->degree; i++)
            if (temp->child[i]->part < 0) return (0);
    }

    return (res);
}
//---------------------------------------------

void Network::SolveImage(long L, double a, double c, bool plain) {
    Queue temp_queue;
    Node *temp;
    int i, j;
    int last_switched_node;

    temp = head;
    while (temp) {
        if (!plain)
            if (temp->through != 2) temp->through = 0;
        temp->visited = temp->degree;
        temp = temp->next_n;
    }

    temp = head;
    while (temp) {
        if (temp->degree == 0) {
            temp->aggregate_p = temp->power;
            temp->aggregate_e = temp->e0;
            temp->parent->visited--;
            if (!temp->parent->visited)
                temp_queue.Insert(temp->parent);
        }
        temp = temp->next_n;
    }

    // Solve the problem. N is temp->degree
    // The first while calculate the aggregate_p and aggregate_e
    while (!temp_queue.IsEmpty()) {
        temp = temp_queue.Pop(); // pops a node ready for aggregate calculations
        ImageAggregate(temp, a, c);
        if (temp->parent) {
            temp->parent->visited--;
            if (!temp->parent->visited)
                temp_queue.Insert(temp->parent);
        }
    }

    // In the second run the actual load each processor gets are calculated
    //temp contains now the root processor which is the last to pop
    // The loop is repeated if recalculation of aggregate values is necessary
    if (!plain) {
        last_switched_node = -1;
        do {
            valid = 1;
            temp = head;
            temp->L = L;
            temp_queue.Insert(temp);
            while (!temp_queue.IsEmpty() && (valid == 1)) {
                temp = temp_queue.Pop();
                if (temp->parent) temp->L = temp->part * temp->parent->L;
                valid = ImagePart(temp, a, c);

                if (valid == 2) {
                    if ((temp->through == 1) && (last_switched_node == temp->ID)) {
                        valid = 1;
                        last_switched_node = -1;
                    } else if (temp->through == 0) temp->through = 1;
                    else temp->through = 0;
                }

                // each child node will be examined next if it is not a leaf
                for (i = temp->degree - 1; i >= 0; i--)
                    if (temp->child[i]->degree) temp_queue.Insert(temp->child[i]);
            }
            if (valid == 2) // aggregate values recalculation
            {
                last_switched_node = temp->ID;
                while (temp) // recalculate up to the top node
                {
                    ImageAggregate(temp, a, c);
                    temp = temp->parent;
                }
                // Empty queue before next iteration
                while (!temp_queue.IsEmpty()) temp_queue.Pop();
            }
        } while (valid == 2);
    } else {
        valid = 1;
        temp = head;
        temp->L = L;
        temp_queue.Insert(temp);
        while (!temp_queue.IsEmpty() && (valid == 1)) {
            temp = temp_queue.Pop();
            if (temp->parent) temp->L = temp->part * temp->parent->L;
            valid = ImagePart(temp, a, c);

            // each child node will be examined next if it is not a leaf
            for (i = temp->degree - 1; i >= 0; i--)
                if (temp->child[i]->degree) temp_queue.Insert(temp->child[i]);
        }
        if (valid == 2) valid = 0;
    }

    // Empty queue in case error occured
    while (!temp_queue.IsEmpty()) temp_queue.Pop();

    // now check T4 optimality condition
    max_gain = 0;
    temp = head;
    t4_holds = 1;
    while (temp) {
        if (temp->degree)
            t4_holds = (t4_holds && CheckT4(temp, a));
        temp = temp->next_n;
        if (!t4_holds)
            max_gain = MIN(max_gain, __prob_gain);
    }
}
//---------------------------------------------

void Network::ImageAggregate(Node *temp, double a, double c) {
    Node *temp1, *temp2;
    double cl, al, aux1, aux2;
    int i, j, k;
    double l; // the link speed assumed for all nodes' connections
    double product[MAX_NODE_DEGREE]; // there are N product values computed in the case of FE and N-2 for NFE
    double s_p, s_s_p;
    double aux;
    double e1, p1, e2, p2;

    l = temp->link[0];
    cl = c*l;
    al = a*l;
    ImageSort(temp);
    if (temp->through) {
        // product[i-1] here holds the product value for j=0,..i-2
        // when i ranges for i=1,..,N
        product[0] = 1;
        if (temp->degree > 1)
            for (i = 1; i < temp->degree; i++)
                product[i] = product[i - 1]*(al + temp->child[i - 1]->aggregate_p) / (cl + temp->child[i]->aggregate_p);

        s_p = 0;
        for (i = 0; i < temp->degree; i++) s_p += product[i];

        s_s_p = 0;
        for (i = 1; i < temp->degree; i++)
            for (j = 0; j <= i - 1; j++) {
                aux = 1;
                if (i > j + 1)
                    aux = product[i] / product[j + 1]; // product from j to i-2
                temp1 = temp->child[j];
                temp2 = temp->child[j + 1];
                p1 = temp1->aggregate_p;
                e1 = temp1->aggregate_e;
                p2 = temp2->aggregate_p;
                e2 = temp2->aggregate_e;
                aux *= (e1 * p1 - e2 * p2) / (p2 + cl);
                s_s_p += aux;
            }
        p1 = temp->child[0]->aggregate_p;
        e1 = temp->child[0]->aggregate_e;
        aux = temp->aggregate_p = al + (p1 + cl) / s_p;
        temp->aggregate_e = (p1 * e1 - (p1 + cl) * s_s_p / s_p) / aux;
    } else {
        if (temp->fe) {
            s_p = 1;
            product[temp->degree - 1] = 1;
            aux1 = aux2 = 1;
            if (temp->degree > 1)
                for (i = temp->degree - 2; i >= 0; i--) {
                    aux1 *= cl + temp->child[i + 1]->aggregate_p;
                    aux2 *= al + temp->child[i]->aggregate_p;
                    product[i] = aux1 / aux2;
                    s_p += aux1 / aux2;
                }

            s_s_p = 0;
            for (i = 1; i <= temp->degree; i++)
                for (j = i + 1; j <= temp->degree; j++) {
                    if (i + 1 <= j - 1)
                        aux = product[i - 1] / product[j - 2];
                    else
                        aux = 1;

                    p1 = temp->child[j - 1]->aggregate_p;
                    e1 = temp->child[j - 1]->aggregate_e;
                    p2 = temp->child[j - 2]->aggregate_p;
                    e2 = temp->child[j - 2]->aggregate_e;
                    aux *= (e1 * p1 - e2 * p2) / (p2 + al);
                    s_s_p += aux;
                }
            aux = 1 + s_p * (temp->power + cl) / (temp->child[temp->degree - 1]->aggregate_p + al);

            p1 = temp->power; // p1 is p0 and p2 is pN
            e1 = temp->e0;
            p2 = temp->child[temp->degree - 1]->aggregate_p;
            e2 = temp->child[temp->degree - 1]->aggregate_e;
            temp->aggregate_p = p1 * (1 + cl * s_p / (al + p2)) / aux;
            temp->aggregate_e = p1 * (e1 - (s_p * (p1 * e1 - p2 * e2) / (p2 + al) + s_s_p) / aux);
            temp->aggregate_e /= temp->aggregate_p;
        } else {
            product[0] = 1;
            if (temp->degree > 1)
                for (i = 1; i < temp->degree; i++)
                    product[i] = product[i - 1]*(al + temp->child[i - 1]->aggregate_p) / (cl + temp->child[i]->aggregate_p);

            s_p = 0;
            for (i = 0; i < temp->degree; i++) s_p += product[i];

            s_s_p = 0;
            for (i = 1; i <= temp->degree; i++)
                for (j = 0; j < i; j++) {
                    aux = 1;
                    if (j <= i - 2) {
                        aux = product[i - 1];
                        if (j > 0) aux /= product[j];
                    }

                    if (j == 0) {
                        p1 = temp->power;
                        e1 = temp->e0;
                    } else {
                        p1 = temp->child[j - 1]->aggregate_p;
                        e1 = temp->child[j - 1]->aggregate_e;
                    }
                    p2 = temp->child[j]->aggregate_p;
                    e2 = temp->child[j]->aggregate_e;
                    aux *= (e1 * p1 - e2 * p2) / (p2 + cl);
                    s_s_p += aux;
                }
            p1 = temp->power;
            e1 = temp->e0;
            p2 = temp->child[0]->aggregate_p;
            e2 = temp->child[0]->aggregate_e;
            aux = 1 + s_p * (p1 - cl) / (p2 + cl);

            temp->aggregate_p = al + cl + (p1 - al - cl)*(1 - cl * s_p / (p2 + cl)) / aux;
            temp->aggregate_e = p1 * e1 - (p1 - al - cl) * s_s_p / aux;
            temp->aggregate_e /= temp->aggregate_p;
        }
    }
}
//---------------------------------------------
// return value 0: invalid net no solution
//              1: ok
//              2: temp node set/reset to 'through' mode.
//'2' response requires recalculation of aggregate values from here to host node

int Network::ImagePart(Node *temp, double a, double c) {
    Node *temp1, *temp2;
    double cl, al;
    int i, j, k;
    double l; // the link speed assumed for all nodes' connections
    double product[MAX_NODE_DEGREE]; // there are N product values computed in the case of FE and N-2 for NFE
    double s_p, s_s_p;
    double aux, aux1, aux2, aux3;
    double e1, p1, e2, p2;
    double min_part[MAX_NODE_DEGREE];
    int res = 1;

    l = temp->link[0];
    cl = c*l;
    al = a*l;

    // check possibility of solution
    if (temp->degree != 1) {
        if (temp->through) min_part[0] = 0;
        else {
            temp1 = temp->child[0];
            p1 = temp1->aggregate_p;
            e1 = temp1->aggregate_e;
            aux1 = temp->power * temp->e0 - p1*e1;
            if (temp->fe)
                min_part[0] = MAX(0, (aux1 - al * temp->L) / (temp->L * (cl + p1)));
            else
                min_part[0] = MAX(0, (aux1 + cl * temp->L) / (temp->L * (cl + p1)));
        }

        for (i = 1; i < temp->degree; i++) {
            temp1 = temp->child[i - 1];
            temp2 = temp->child[i];
            aux1 = temp1->aggregate_p * temp1->aggregate_e - temp2->aggregate_p * temp2->aggregate_e;
            aux2 = (cl + temp2->aggregate_p) * temp->L;
            min_part[i] = MAX(0, (al * min_part[i - 1] * temp->L + aux1) / aux2);
        }
        aux1 = 0;
        for (i = 0; i < temp->degree; i++) aux1 += min_part[i];
        if (aux1 > 1) return (0);
    }

    // Solution regardless of 'through' state
    if (temp->fe) {
        s_p = 1;
        product[temp->degree - 1] = 1;
        aux1 = aux2 = 1;
        if (temp->degree > 1)
            for (i = temp->degree - 2; i >= 0; i--) {
                aux1 *= cl + temp->child[i + 1]->aggregate_p;
                aux2 *= al + temp->child[i]->aggregate_p;
                product[i] = aux1 / aux2;
                s_p += aux1 / aux2;
            }

        s_s_p = 0;
        for (i = 1; i <= temp->degree; i++)
            for (j = i + 1; j <= temp->degree; j++) {
                if (i + 1 <= j - 1)
                    aux = product[i - 1] / product[j - 2];
                else
                    aux = 1;

                p1 = temp->child[j - 1]->aggregate_p;
                e1 = temp->child[j - 1]->aggregate_e;
                p2 = temp->child[j - 2]->aggregate_p;
                e2 = temp->child[j - 2]->aggregate_e;
                aux *= (e1 * p1 - e2 * p2) / (p2 + al);
                s_s_p += aux;
            }
        aux = 1 + s_p * (temp->power + cl) / (temp->child[temp->degree - 1]->aggregate_p + al);

        p1 = temp->power; // p1 is p0 and p2 is pN
        e1 = temp->e0;
        p2 = temp->child[temp->degree - 1]->aggregate_p;
        e2 = temp->child[temp->degree - 1]->aggregate_e;

        temp->part = 1 - s_p * ((p1 * e1 - p2 * e2) / temp->L - cl) / (p2 + al) - s_s_p / temp->L;
        temp->part /= aux;


        // now calculate for each child parti starting from partN
        temp2 = temp->child[temp->degree - 1];
        temp2->part = (p1 * e1 - p2 * e2) / temp->L - cl + (p1 + cl) * temp->part;
        temp2->part /= (p2 + al);
        for (i = temp->degree - 2; i >= 0; i--) // i ranges 1..N
        {
            temp2 = temp->child[i];
            p1 = p2;
            e1 = e2;
            p2 = temp2->aggregate_p;
            e2 = temp2->aggregate_e;
            temp2->part = (p1 * e1 - p2 * e2) / temp->L + (p1 + cl) * temp->child[i + 1]->part;
            temp2->part /= (p2 + al);
        }
    } else {
        product[0] = 1;
        if (temp->degree > 1)
            for (i = 1; i < temp->degree; i++)
                product[i] = product[i - 1]*(al + temp->child[i - 1]->aggregate_p) / (cl + temp->child[i]->aggregate_p);

        s_p = 0;
        for (i = 0; i < temp->degree; i++) s_p += product[i];

        s_s_p = 0;
        for (i = 1; i <= temp->degree; i++)
            for (j = 0; j < i; j++) {
                aux = 1;
                if (j <= i - 2) {
                    aux = product[i - 1];
                    if (j > 0) aux /= product[j];
                }

                if (j == 0) {
                    p1 = temp->power;
                    e1 = temp->e0;
                } else {
                    p1 = temp->child[j - 1]->aggregate_p;
                    e1 = temp->child[j - 1]->aggregate_e;
                }
                p2 = temp->child[j]->aggregate_p;
                e2 = temp->child[j]->aggregate_e;
                aux *= (e1 * p1 - e2 * p2) / (p2 + cl);
                s_s_p += aux;
            }
        p1 = temp->power;
        p2 = temp->child[0]->aggregate_p;
        aux = 1 + (p1 - cl) * s_p / (p2 + cl);
        temp->part = (1 - s_s_p / temp->L - cl * s_p / (p2 + cl)) / aux;

        // now calculate for each child parti
        e1 = temp->e0;
        e2 = temp->child[0]->aggregate_e;
        temp->child[0]->part = (p1 * e1 - e2 * p2) / temp->L + cl + (p1 - cl) * temp->part;
        temp->child[0]->part /= (p2 + cl);

        for (i = 1; i < temp->degree; i++) // i ranges 2..N
        {
            p1 = p2;
            e1 = e2;
            p2 = temp->child[i]->aggregate_p;
            e2 = temp->child[i]->aggregate_e;
            temp->child[i]->part = (p1 * e1 - p2 * e2) / temp->L + (p1 + al) * temp->child[i - 1]->part;
            temp->child[i]->part /= (p2 + cl);
        }
    }


    // check validity of solution
    for (i = 0; i < temp->degree; i++)
        if (temp->child[i]->part < 0) return (0);
    if ((temp->part < 0) && (temp->through == 0)) // set the 'through' flag
    {
        //    temp->through=1;
        res = 2;
    } else if ((temp->part > 0) && (temp->through == 1)) // reset the 'through' flag
    {
        //     temp->through=0;
        res = 2;
    }


    // now solve again
    if (temp->through) // host is not participating
    {
        // product[i-1] here holds the product value for j=0,..i-2
        // when i ranges for i=1,..,N
        product[0] = 1;
        if (temp->degree > 1)
            for (i = 1; i < temp->degree; i++)
                product[i] = product[i - 1]*(al + temp->child[i - 1]->aggregate_p) / (cl + temp->child[i]->aggregate_p);

        s_p = 0;
        for (i = 0; i < temp->degree; i++) s_p += product[i];

        s_s_p = 0;
        for (i = 1; i < temp->degree; i++)
            for (j = 0; j <= i - 1; j++) {
                aux = 1;
                if (i > j + 1)
                    aux = product[i] / product[j + 1]; // product from j to i-2
                temp1 = temp->child[j];
                temp2 = temp->child[j + 1];
                p1 = temp1->aggregate_p;
                e1 = temp1->aggregate_e;
                p2 = temp2->aggregate_p;
                e2 = temp2->aggregate_e;
                aux *= (e1 * p1 - e2 * p2) / (p2 + cl);
                s_s_p += aux;
            }
        s_s_p /= temp->L;

        // this is part1, part0 is 0
        temp->part = 0;
        temp1 = temp->child[0];
        aux1 = temp1->part = (1 - s_s_p) / s_p;

        // now calculate for each child parti with the recursive formula which
        // proved more numerically stable
        p2 = temp->child[0]->aggregate_p;
        e2 = temp->child[0]->aggregate_e;
        for (i = 1; i < temp->degree; i++) // i ranges 2..N
        {
            p1 = p2;
            e1 = e2;
            p2 = temp->child[i]->aggregate_p;
            e2 = temp->child[i]->aggregate_e;
            temp->child[i]->part = (p1 * e1 - p2 * e2) / temp->L + (p1 + al) * temp->child[i - 1]->part;
            temp->child[i]->part /= (p2 * cl);
        }
    }

    aux3 = temp->part;
    for (i = 0; i < temp->degree; i++) {
        if (temp->child[i]->part < 0) return (0);
        aux3 += temp->child[i]->part;
    }
    if (fabs(aux3 - 1.0) > 0.0001) /* needed because of lack of numerical stability */
        return (0);
    else return (res);
}
//---------------------------------------------
// Can be called only after a SolveImage method

void Network::PrintTimes4Image(double a, double c) {
    Queue node_queue;
    Node *temp;
    double STR, R, C, S; // start to receive load, received, collected, sent back results
    int i, j;
    double l;

    temp = head;
    l = head->link[0];
    temp->time = 0;
    node_queue.Insert(temp);
    printf("Name      STR       R         C        S\n");
    while (!node_queue.IsEmpty()) {
        temp = node_queue.Pop();
        STR = temp->time;
        if (temp->parent) {
            if (temp->degree) {
                R = STR + temp->L * l*a;
                C = R + temp->aggregate_p * (temp->L + temp->aggregate_e);
                S = C + l * c * temp->L;
            } else {
                R = STR + temp->parent->L * temp->part * l*a;
                C = R + temp->power * (temp->e0 + temp->parent->L * temp->part);
                S = C + temp->parent->L * temp->part * c*l;
            }
        } else // only for host node
        {
            R = 0;
            C = S = temp->aggregate_p * (temp->L + temp->aggregate_e);
        }
        printf("%s %f %f %f %f\n", temp->name, STR, R, C, S);
        for (i = temp->degree - 1; i >= 0; i--) {
            temp->child[i]->time = R;
            node_queue.Insert(temp->child[i]);
            if (temp->child[i]->degree)
                R += a * l * temp->child[i]->L;
            else
                R += a * l * temp->child[i]->part * temp->L;
        }
    }
}
//---------------------------------------------

void Network::PrintTimes4Query(double b, double d) {
    Queue node_queue;
    Node *temp;
    double STR, R, C, S; // start to receive load, received, collected, sent back results
    int i, j;
    double l;

    temp = head;
    temp->time = 0;
    node_queue.Insert(temp);
    printf("Name      STR       R         C        S\n");
    while (!node_queue.IsEmpty()) {
        temp = node_queue.Pop();
        STR = temp->time;
        if (temp->parent) {
            l = temp->l2par;

            R = STR + l*b;
            if (temp->degree)
                C = R + temp->aggregate_p * (temp->L + temp->aggregate_e);
            else
                C = R + temp->power * (temp->e0 + temp->parent->L * temp->part);
            S = C + l*d;
        } else // only for host node
        {
            R = 0;
            C = S = temp->aggregate_p * (temp->L + temp->aggregate_e);
        }
        printf("%s %f %f %f %f\n", temp->name, STR, R, C, S);
        for (i = 0; i < temp->degree; i++) {
            temp->child[i]->time = R;
            node_queue.Insert(temp->child[i]);
            R += b * temp->link[i];
        }
    }
}
//**************************************************************

double Network::SimulateQuery(double b, double d, bool output) {
    Queue node_queue;
    Node *temp;
    int i, j;
    double l, aux;

    temp = head;
    do {
        temp->visited = temp->degree;
    } while (temp = temp->next_n);


    // initial distribution
    temp = head;
    temp->t.STR = temp->t.FR = 0;
    node_queue.Insert(temp);
    while (!node_queue.IsEmpty()) {
        temp = node_queue.Pop();
        for (i = 0; i < temp->degree; i++) {
            if (!i) aux = temp->child[0]->t.STR = temp->t.FR;
            else temp->child[i]->t.STR = aux;
            aux = temp->child[i]->t.FR = aux + b * temp->link[i];
            if (temp->child[i]->degree) node_queue.Insert(temp->child[i]);
            else temp->child[i]->t.STP = aux;
        }
        if (temp->fe) temp->t.STP = temp->t.FR;
        else temp->t.STP = aux;
    }

    // processing stage
    temp = head;
    do {
        if (temp->degree) {
            if (temp->part != 0)
                temp->t.FPRO = temp->t.STP + temp->power * (temp->L * temp->part + temp->e0);
            else
                temp->t.FPRO = temp->t.STP;
        } else {
            temp->t.RTS = temp->t.FPRO = temp->t.STP + temp->power * (temp->parent->L * temp->part + temp->e0);
            if (!(--temp->parent->visited)) node_queue.Insert(temp->parent);
        }
    } while (temp = temp->next_n);

    // gathering stage
    while (!node_queue.IsEmpty()) {
        temp = node_queue.Pop();
        if (temp->fe) {
            for (i = temp->degree - 1; i >= 0; i--) {
                if (i == temp->degree - 1) aux = temp->child[i]->t.STS = temp->child[i]->t.RTS;
                else aux = temp->child[i]->t.STS = MAX(aux, temp->child[i]->t.RTS);
                aux = temp->child[i]->t.FS = aux + d * temp->link[i];
            }
            temp->t.RTS = MAX(aux, temp->t.FPRO);
        } else {
            for (i = temp->degree - 1; i >= 0; i--) {
                if (i == temp->degree - 1) aux = temp->child[i]->t.STS = MAX(temp->t.FPRO, temp->child[i]->t.RTS);
                else aux = temp->child[i]->t.STS = MAX(aux, temp->child[i]->t.RTS);
                aux = temp->child[i]->t.FS = aux + d * temp->link[i];
            }
            temp->t.RTS = aux;
        }
        if (temp != head) // host has no parent
            if (!(--temp->parent->visited)) node_queue.Insert(temp->parent);
    }
    temp->t.STS = temp->t.FS = temp->t.RTS;

    // printing and cleanup
    if (output) {
        printf("Name      STR       FR        STP      FPRO      RTS        STS       FS\n");
        temp = head;
        do {
            printf("%s   %f  %f  %f  %f  %f  %f  %f\n", temp->name, temp->t.STR, temp->t.FR, temp->t.STP, temp->t.FPRO
                    , temp->t.RTS, temp->t.STS, temp->t.FS);
        } while (temp = temp->next_n);
    }
    aux = head->t.FS;
    return (aux);
}
//**************************************************************

double Network::SimulateImage(double a, double c, bool output) {
    Queue node_queue;
    Node *temp;
    int i, j;
    double l, aux;

    temp = head;
    do {
        temp->visited = temp->degree;
    } while (temp = temp->next_n);


    // initial distribution
    temp = head;
    temp->t.STR = temp->t.FR = 0;
    node_queue.Insert(temp);
    while (!node_queue.IsEmpty()) {
        temp = node_queue.Pop();
        for (i = temp->degree - 1; i >= 0; i--) {
            if (i == temp->degree - 1) aux = temp->child[i]->t.STR = temp->t.FR;
            else temp->child[i]->t.STR = aux;
            if (temp->child[i]->degree)
                aux = temp->child[i]->t.FR = aux + a * temp->child[i]->L * temp->link[i];
            else
                aux = temp->child[i]->t.FR = aux + a * temp->L * temp->child[i]->part * temp->link[i];
            if (temp->child[i]->degree) node_queue.Insert(temp->child[i]);
            else temp->child[i]->t.STP = aux;
        }
        if (temp->fe) temp->t.STP = temp->t.FR;
        else temp->t.STP = aux;
    }

    // processing stage
    temp = head;
    do {
        if (temp->degree) {
            if (temp->part != 0)
                temp->t.FPRO = temp->t.STP + temp->power * (temp->L * temp->part + temp->e0);
            else
                temp->t.FPRO = temp->t.STP;
        } else {
            temp->t.RTS = temp->t.FPRO = temp->t.STP + temp->power * (temp->parent->L * temp->part + temp->e0);
            if (!(--temp->parent->visited)) node_queue.Insert(temp->parent);
        }
    } while (temp = temp->next_n);

    // gathering stage
    while (!node_queue.IsEmpty()) {
        temp = node_queue.Pop();
        if (temp->fe) {
            for (i = temp->degree - 1; i >= 0; i--) {
                if (i == temp->degree - 1) aux = temp->child[i]->t.STS = temp->child[i]->t.RTS;
                else aux = temp->child[i]->t.STS = MAX(aux, temp->child[i]->t.RTS);
                if (temp->child[i]->degree)
                    aux = temp->child[i]->t.FS = aux + c * temp->child[i]->L * temp->link[i];
                else
                    aux = temp->child[i]->t.FS = aux + c * temp->L * temp->child[i]->part * temp->link[i];
            }
            temp->t.RTS = MAX(aux, temp->t.FPRO);
        } else {
            for (i = temp->degree - 1; i >= 0; i--) {
                if (i == temp->degree - 1) aux = temp->child[i]->t.STS = MAX(temp->t.FPRO, temp->child[i]->t.RTS);
                else aux = temp->child[i]->t.STS = MAX(aux, temp->child[i]->t.RTS);
                if (temp->child[i]->degree)
                    aux = temp->child[i]->t.FS = aux + c * temp->child[i]->L * temp->link[i];
                else
                    aux = temp->child[i]->t.FS = aux + c * temp->L * temp->child[i]->part * temp->link[i];
            }
            temp->t.RTS = aux;
        }
        if (temp != head) // host has no parent
            if (!(--temp->parent->visited)) node_queue.Insert(temp->parent);
    }
    temp->t.STS = temp->t.FS = temp->t.RTS;

    // printing and cleanup
    if (output) {
        printf("Name      STR       FR        STP      FPRO      RTS        STS       FS\n");
        temp = head;
        do {
            printf("%s   %f  %f  %f  %f  %f  %f  %f\n", temp->name, temp->t.STR, temp->t.FR, temp->t.STP, temp->t.FPRO
                    , temp->t.RTS, temp->t.STS, temp->t.FS);
        } while (temp = temp->next_n);
    }

    aux = head->t.FS;
    return (aux);
}
//**************************************************************
// The tail pointer is not updated so that the previous state
// can be removed

void Network::InsertDuplicateNode(Node *x) {
    Node *y = x->parent;

    if (!y) // only for host node
    {
        head = tail = x;
    } else {
        tail->next_n = x;
        y->link[y->degree] = x->l2par;
        y->child[y->degree++] = x;
        while (!y->next_n) // insert parent if not already inside net
        {
            x->next_n = y;
            x = y;
            y = y->parent;
            y->link[y->degree] = x->l2par;
            y->child[y->degree++] = x;
        }
    }
}
//**************************************************************
// Removes all duplicate nodes appended for testing

void Network::RemoveDuplicateNode(void) {
    Node *x, *y = tail;
    int i;

    x = y->next_n;
    do {
        y->next_n = NULL; // signal that it is not active
        i = 0;
        while (x->parent->child[i] != x) i++;
        x->parent->link[i] = x->parent->link[--(x->parent->degree)];
        x->parent->child[i] = x->parent->child[x->parent->degree];
        y = x;
        x = y->next_n;
    } while (x);
}
//**************************************************************
// Creates a duplicate node for greedy methods

Node *DuplicateNode(Node *x) {
    Node *temp;
    temp = new(Node);
    strcpy(temp->name, x->name);
    temp->ID = x->ID;
    temp->power = x->power;
    temp->next_n = NULL;
    temp->e0 = x->e0;
    temp->degree = 0;
    temp->l2par = x->l2par;
    temp->visited = 0;
    temp->fe = x->fe;
    temp->parent = NULL;
    temp->through = x->through;
    return (temp);
}
//**************************************************************
// The best performing subnet is returned

void Network::Aux2Greedy(bool ImageOrQuery, Network &test, long L, double ab, double cd) {
    Queue temp_queue;
    Node *x, *y; // holds the newly inserted node
    unsigned int in_array = 0, num_nodes;
    int i, j, k;
    double min_time, temp_time, local_min;
    int best_i; // index of best newly inserted node. -1 for no insertion
    long lab = (long) ab, lcd = (long) cd;
    int leafs;
    Node *first = NULL;

    test.improved_flag = 0;

    // this step sorts the nodes before attempting a solution 
    // because uniform machines can lead to local minima that result in errors
    test = *this;
    if (ImageOrQuery) test.SolveImage(INT_MAX, ab, cd);
    else test.SolveQuery(INT_MAX, lab, lcd);

    num_nodes = strlen(test.node_usage);
    in_array = num_nodes - 1;

    // Fake the removal of all nodes except host node
    for (i = 0; i < in_array; i++) {
        nodes_array[i] = &test.netnode[i + 1];
        test.netnode[i + 1].degree = 0;
        test.netnode[i + 1].next_n = NULL;
    }
    test.head->degree = 0;
    test.tail = test.head;


    if (head->through) min_time = FLT_MAX;
    else min_time = test.head->power * (L + test.head->e0);
    do {
        best_i = -1;
        for (i = 0; i < in_array; i++) {
            test.InsertDuplicateNode(nodes_array[i]);
            if (ImageOrQuery) test.SolveImage(L, ab, cd);
            else test.SolveQuery(L, lab, lcd);
            if (test.valid == 1) {
                temp_time = test.head->aggregate_p * (L + test.head->aggregate_e);
                if (temp_time < min_time) {
                    min_time = temp_time;
                    best_i = i;
                }
            }
            test.RemoveDuplicateNode();
        }
        if (best_i >= 0) {
            if (!first) first = nodes_array[best_i];
            test.InsertDuplicateNode(nodes_array[best_i]);
            x = test.head;
            while (x->next_n) x = x->next_n;
            test.tail = x;

            i = 0;
            do {
                if (!nodes_array[i]->next_n && (test.tail != nodes_array[i])) i++;
                else nodes_array[i] = nodes_array[--in_array];
            } while (i < in_array);
        }
    } while (in_array && (best_i >= 0)); // if any nodes left for addition

    // try improving solution
    if (first && in_array) {
        for (i = 0; i < num_nodes; i++) {
            if ((test.netnode[i].next_n) || (test.tail == &test.netnode[i]))
                usage_examined[i] = 1;
            else
                usage_examined[i] = 0;
        }

        // test all the tree
        if (ImageOrQuery) SolveImage(L, ab, cd);
        else SolveQuery(L, lab, lcd);
        temp_time = head->aggregate_p * (L + head->aggregate_e);
        if ((valid == 1) && (temp_time < min_time)) {
            min_time = temp_time;
            memset(usage_best, 1, num_nodes);
        } else
            memcpy(usage_best, usage_examined, num_nodes);

        leafs = 0;
        x = test.head->next_n;
        while (x) {
            //      if(x->parent->next_n || (x->parent->degree>1))
            leaf_array[leafs++] = x;
            x = x->next_n;
        }

        // destroy net
        for (j = 0; j < num_nodes; j++) {
            test.netnode[j].next_n = NULL;
            test.netnode[j].degree = 0;
        }

        for (i = 0; i < leafs && leafs > 1; i++) // if leafs==1 this loop will not produce a better solution
        {
            // restore initial solution
            x = test.head;
            for (j = 1; j < num_nodes; j++)
                if (usage_examined[j]) {
                    x->next_n = &test.netnode[j];
                    x = &test.netnode[j];
                    y = x->parent;
                    y->link[y->degree] = x->l2par;
                    y->child[y->degree++] = x;
                }
            // cut off the leaf examined along with its unique path to the root
            // unless it is an internal node
            x = leaf_array[i];
            if (x->degree) {
                j = 0;
                while (x->parent->child[j] != x) j++;
                x->parent->link[j] = x->parent->link[--(x->parent->degree)];
                x->parent->child[j] = x->parent->child[x->parent->degree];
                temp_queue.Insert(leaf_array[i]);
                while (!temp_queue.IsEmpty()) {
                    x = temp_queue.Pop();
                    for (j = 0; j < x->degree; j++)
                        temp_queue.Insert(x->child[j]);
                    y = &test.netnode[0];
                    while (y->next_n != x) y = y->next_n;
                    y->next_n = x->next_n;
                    x->next_n = NULL;
                    x->degree = 0;
                }
            } else
                do {
                    y = &test.netnode[0];
                    while (y->next_n != x) y = y->next_n;
                    y->next_n = x->next_n;
                    x->next_n = NULL;
                    j = 0;
                    while (x->parent->child[j] != x) j++;
                    x->parent->link[j] = x->parent->link[--(x->parent->degree)];
                    x->parent->child[j] = x->parent->child[x->parent->degree];
                    x = x->parent;
                } while (x && (x->degree == 0) && (x->parent));

            x = test.head;
            while (x->next_n) x = x->next_n;
            test.tail = x;

            in_array = 0;
            for (j = 0; j < num_nodes; j++)
                if (!(test.netnode[j].next_n) &&
                        (test.tail != &test.netnode[j]) &&
                        (leaf_array[i] != &test.netnode[j]))
                    nodes_array[in_array++] = &test.netnode[j];

            if (test.head->next_n) {
                if (ImageOrQuery) test.SolveImage(L, ab, cd);
                else test.SolveQuery(L, lab, lcd);
                if (test.valid == 1)
                    local_min = test.head->aggregate_p * (L + test.head->aggregate_e);
                else
                    local_min = MAX_FLOAT;

                do {
                    best_i = -1;
                    for (k = 0; k < in_array; k++) {
                        test.InsertDuplicateNode(nodes_array[k]);
                        if (ImageOrQuery) test.SolveImage(L, ab, cd);
                        else test.SolveQuery(L, lab, lcd);
                        if (test.valid == 1) {
                            temp_time = test.head->aggregate_p * (L + test.head->aggregate_e);
                            if (temp_time < local_min) {
                                local_min = temp_time;
                                best_i = k;
                            }
                        }
                        test.RemoveDuplicateNode();
                    }
                    if (best_i >= 0) {
                        test.InsertDuplicateNode(nodes_array[best_i]);
                        test.tail->next_n = nodes_array[best_i];
                        while (test.tail->next_n) test.tail = test.tail->next_n;

                        j = 0;
                        do {
                            if (!nodes_array[j]->next_n && (test.tail != nodes_array[j])) j++;
                            else nodes_array[j] = nodes_array[--in_array];
                        } while (j < in_array);

                    }
                } while (in_array && (best_i >= 0)); // if any nodes left for addition

                if (local_min < min_time) {
                    test.improved_flag = 1;
                    min_time = local_min;
                    for (j = 0; j < num_nodes; j++) {
                        if ((test.netnode[j].next_n) || (test.tail == &test.netnode[j]))
                            usage_best[j] = 1;
                        else
                            usage_best[j] = 0;
                    }
                }

            }

            // destroy net
            for (j = 0; j < num_nodes; j++) {
                test.netnode[j].next_n = NULL;
                test.netnode[j].degree = 0;
            }
        }

        x = test.head;
        for (i = 1; i < num_nodes; i++)
            if (usage_best[i]) {
                x->next_n = &test.netnode[i];
                x = &test.netnode[i];
                y = x->parent;
                y->link[y->degree] = x->l2par;
                y->child[y->degree++] = x;
                test.tail = x;
            } else test.node_usage[i] = 0;
    }

    if (test.head != test.tail)
        if (ImageOrQuery) test.SolveImage(L, ab, cd);
        else test.SolveQuery(L, lab, lcd);
}
//**************************************************************
void Network::GreedyQueryRev(Network &test, long L, double b, double d) {
  this->GreedyRev(0, test, L, b, d);  
}
//**************************************************************
void Network::GreedyImageRev(Network &test, long L, double a, double c) {
  this->GreedyRev(1, test, L, a, c);
}
//**************************************************************
// Greedy approach that start from the reverse of Aux2Greedy
// It progresses by removing the slower leaf nodes.

void Network::GreedyRev(bool ImageOrQuery, Network &test, long L, double ab, double cd) {
    Queue temp_queue;
    Node *x, *y, *z; // holds the newly inserted node
    int i, j;
    double min_time, temp_time, aux;
    double min_pow;
    int best_i; // index of best newly inserted node. -1 for no insertion
    long lab = (long) ab, lcd = (long) cd;
    int leafs;
    int examined;

    test = *this;
    if (ImageOrQuery) test.SolveImage(L, ab, cd);
    else test.SolveQuery(L, lab, lcd);

    if (test.valid == 1) min_time = test.head->aggregate_p * (L + test.head->aggregate_e);
    else min_time = MAX_FLOAT;

    examined = 0;
    do {
        x = test.head->next_n;
        leafs = 0;
        while (x) {
            if (!x->degree) nodes_array[leafs++] = x;
            x = x->next_n;
        }
        best_i = -1;
        for (i = 0; i < leafs; i++) {
            // remove leaf node
            x = nodes_array[i];
            y = &test.netnode[0];
            while (y->next_n != x) y = y->next_n;
            y->next_n = x->next_n;
            j = 0;
            while (x->parent->child[j] != x) j++;
            x->parent->link[j] = x->parent->link[--(x->parent->degree)];
            x->parent->child[j] = x->parent->child[x->parent->degree];

            if (test.head->next_n)
                if (ImageOrQuery) test.SolveImage(L, ab, cd);
                else test.SolveQuery(L, lab, lcd);
            else {
                test.valid = 1;
                test.netnode[0].aggregate_p = test.netnode[0].power;
                test.netnode[0].aggregate_e = test.netnode[0].e0;
            }

            if (test.valid == 1) {
                examined++;
                temp_time = test.head->aggregate_p * (L + test.head->aggregate_e);
                if (temp_time < min_time) {
                    min_time = temp_time;
                    best_i = i;
                }
            }

            // re-insert leaf node
            x->next_n = y->next_n;
            y->next_n = x;
            x->parent->link[x->parent->degree] = x->l2par;
            x->parent->child[x->parent->degree] = x;
            x->parent->degree++;
        }

        if ((best_i < 0) && !examined) // find what to remove
        {
            best_i = 0;
            aux = 0;
            z = nodes_array[0];
            while (z->parent) {
                aux += z->l2par;
                z = z->parent;
            }
            min_pow = nodes_array[0]->power*aux;
            for (i = 1; i < leafs; i++) {
                aux = 0;
                z = nodes_array[0];
                while (z->parent) {
                    aux += z->l2par;
                    z = z->parent;
                }
                if (min_pow < nodes_array[i]->power * aux) {
                    best_i = i;
                    min_pow = nodes_array[i]->power*aux;
                }
            }
        }
        if (best_i >= 0) {
            // remove leaf node
            x = nodes_array[best_i];
            y = &test.netnode[0];
            while (y->next_n != x) y = y->next_n;
            y->next_n = x->next_n;
            x->next_n = NULL;
            j = 0;
            while (x->parent->child[j] != x) j++;
            x->parent->link[j] = x->parent->link[--(x->parent->degree)];
            x->parent->child[j] = x->parent->child[x->parent->degree];
        }
    } while (best_i >= 0);

    x = test.head;
    while (x->next_n) x = x->next_n;
    test.tail = x;

    if (test.head->next_n)
        if (ImageOrQuery) test.SolveImage(L, ab, cd);
        else test.SolveQuery(L, lab, lcd);
}
//**************************************************************
// The visited flag in the original network tree nodes is used to
// signal that a node has already been inserted in the test tree
// An array is used to hold pointers to the duplicate nodes.
// In order to limit the search to uninserted nodes,
// we send the newly inserted to the end of the array

void Network::AuxGreedy(bool ImageOrQuery, long start_L, long end_L, double ab, double cd) {
    Network test;
    Node *x; // holds the newly inserted node
    Node **nodes_array;
    unsigned int num_nodes = 1;
    unsigned int in_array = 0;
    int i;
    long temp_L;
    double min_time, temp_time;
    int best_i; // index of best newly inserted node. -1 for no insertion
    long lab = (long) ab, lcd = (long) cd;

    // this step sorts the nodes before attempting a solution 
    // because uniform machines can lead to local minima that result in errors
    if (ImageOrQuery) SolveImage(INT_MAX, ab, cd);
    else SolveQuery(INT_MAX, lab, lcd);

    x = head;
    while (x = x->next_n) num_nodes++;

    nodes_array = (Node **) calloc(num_nodes, sizeof (Node *));
    x = head;
    do {
        nodes_array[in_array] = DuplicateNode(x);
        i = 0;
        while (i < in_array) {
            if (!strcmp(nodes_array[i]->name, x->parent->name)) {
                nodes_array[in_array]->parent = nodes_array[i];
                i = in_array;
            }
            i++;
        }
        in_array++;
    } while (x = x->next_n);

    test.InsertDuplicateNode(nodes_array[0]);
    nodes_array[0] = nodes_array[--in_array];

    for (temp_L = start_L; temp_L <= end_L; temp_L++) {
        if (test.head == test.tail) // only host node present
            min_time = test.head->power * (temp_L + test.head->e0);
        else {
            if (ImageOrQuery) test.SolveImage(temp_L, ab, cd);
            else test.SolveQuery(temp_L, lab, lcd);
            min_time = test.head->aggregate_p * (temp_L + test.head->aggregate_e);
        }
        do {
            best_i = -1;
            for (i = 0; i < in_array; i++) {
                test.InsertDuplicateNode(nodes_array[i]);
                if (ImageOrQuery) test.SolveImage(temp_L, ab, cd);
                else test.SolveQuery(temp_L, lab, lcd);

                temp_time = test.head->aggregate_p * (temp_L + test.head->aggregate_e);
                if ((temp_time < min_time) && test.valid) {
                    min_time = temp_time;
                    best_i = i;
                }
                test.RemoveDuplicateNode();
            }
            if (best_i >= 0) {
                test.InsertDuplicateNode(nodes_array[best_i]);
                x = test.tail->next_n;
                while (test.tail->next_n) test.tail = test.tail->next_n;
                i = 0;
                do {
                    if (!nodes_array[i]->next_n && (test.tail != nodes_array[i])) i++;
                    else nodes_array[i] = nodes_array[--in_array];
                } while (i < in_array);
            }
        } while (in_array && (best_i >= 0)); // if any nodes left for addition
        if (test.head == test.tail) // only host node present
        {
            test.head->L = temp_L;
            printf("Processor %s gets %li\n", test.head->name, temp_L);
        } else {
            if (ImageOrQuery) test.SolveImage(temp_L, ab, cd);
            else test.SolveQuery(temp_L, lab, lcd);
            /*          test.Quantify();
               test.PrintSolution(1);
            if(temp_L==100) test.SimulateQuery(lab,lcd,1);
               test.Propagate();
            if(temp_L==100) test.SimulateQuery(lab,lcd,1);
                      test.PrintSolution(1);
             */
        }
    }
    test.~Network();
    free(nodes_array);
}
//**************************************************************
// ImageOrQuery flag differentiates between the two cases
// All the possible usage patterns are examined. They sum up
// to a total of 2^num_nodes but only a subset is valid and
// checked for timing
// Returns the optimum subnet as a duplicate independent net.
// The usage patterns are generated by binary addition! An extra member of array usage_examined
// acts as a carry flag which will be set when all patterns have been examined.
// usage_examined array extends from the processors closest to host to the more distant ones.

void Network::AuxOptimum(bool ImageOrQuery, Network &test, long L, double ab, double cd) {
    Node *x, *y; // holds the newly inserted node
    unsigned int num_nodes; // number of nodes appart from root
    int i, j;
    double min_time, temp_time;
    char carry;
    unsigned long num_patterns = 0;
    long lab = (long) ab, lcd = (long) cd;
    int conductors = 0;
    char no_cond;

    test = *this;
    num_nodes = strlen(test.node_usage);
    for (i = 0; i < num_nodes; i++)
        if (test.netnode[i].degree && (test.netnode[i].through != 2)) conductors++;

    // Fake the removal of all nodes except host node
    for (i = 0; i < num_nodes; i++) {
        nodes_array[i] = &test.netnode[i + 1];
        test.netnode[i + 1].degree = 0;
        test.netnode[i + 1].next_n = NULL;
    }
    test.head->degree = 0;
    test.tail = test.head;

    num_nodes--; // number of nodes appart from root
    for (i = 0; i < conductors; i++) cond_best[i] = 0;
    for (i = 0; i < num_nodes; i++) usage_best[i] = usage_examined[i] = 0;
    usage_examined[num_nodes] = 0; // flag for search termination

    if (head->through != 2)
        min_time = test.head->power * (L + test.head->e0);
    else
        min_time = FLT_MAX;

    do {
        //generate new pattern
        i = 0;
        carry = 0;
        usage_examined[0] += 1;
        do {
            if (usage_examined[i] > 1) {
                usage_examined[i] = 0;
                carry = 1;
            } else carry = 0;
            i++;
            usage_examined[i] += carry;
        } while (carry);

        // check validity of pattern. 'carry' carries the acceptance or not
        // visited flag is used for checking the state of parent nodes
        carry = 1;
        for (i = 1; i <= num_nodes; i++) test.netnode[i].visited = usage_examined[i - 1];
        for (i = 1; i <= num_nodes; i++)
            if (usage_examined[i - 1])
                if (test.netnode[i].parent != test.head)
                    if (!test.netnode[i].parent->visited) {
                        carry = 0;
                        i = num_nodes;
                    }

        if (carry && !usage_examined[num_nodes]) {
            x = test.tail;
            for (i = 1; i <= num_nodes; i++)
                if (usage_examined[i - 1]) {
                    x->next_n = &test.netnode[i];
                    x = &test.netnode[i];
                    y = x->parent;
                    y->link[y->degree] = x->l2par;
                    y->child[y->degree++] = x;
                }

            // test all possible conductors for this active subnet
            no_cond = 1; // first entry
            conductors = 0;
            x = test.head;
            do
                if (x->degree && (x->through != 2)) is_cond[conductors++] = x;
                while (x = x->next_n);
            for (i = 0; i <= conductors; i++) cond_pattern[i] = 0; // last element is a flag
            while (!cond_pattern[conductors]) // until overflow
            {
                num_patterns++;
                for (i = 0; i < conductors; i++)
                    if (is_cond[i]->through != 2) is_cond[i]->through = cond_pattern[i];

                if (ImageOrQuery) test.SolveImage(L, ab, cd, 1);
                else test.SolveQuery(L, lab, lcd, 1);

                if (test.valid == 1) {
                    temp_time = test.head->aggregate_p * (L + test.head->aggregate_e);
                    if (temp_time < min_time) {
                        min_time = temp_time;
                        for (i = 0; i < num_nodes; i++) usage_best[i] = usage_examined[i];
                        memcpy(cond_best, cond_pattern, sizeof (unsigned char)*conductors);
                    }
                }

                // generate new conductor pattern
                i = 0;
                carry = 1;
                do {
                    cond_pattern[i] += carry;
                    if (cond_pattern[i] > 1) {
                        cond_pattern[i] = 0;
                        carry = 1;
                    } else carry = 0;
                    i++;
                } while (carry);
            }

            // now restore net
            x = test.head;
            y = x->next_n;
            do {
                x->next_n = NULL;
                x = y;
                i = 0;
                while (x->parent->child[i] != x) i++;
                x->parent->link[i] = x->parent->link[--(x->parent->degree)];
                x->parent->child[i] = x->parent->child[x->parent->degree];
                y = x->next_n;
            } while (y);
        }
    } while (!usage_examined[num_nodes]); // until carry flag set
    //  printf("Total patterns examined %li\n",num_patterns);
    x = test.tail;
    for (i = 1; i <= num_nodes; i++)
        if (usage_best[i - 1]) {
            x->next_n = &test.netnode[i];
            x = &test.netnode[i];
            y = x->parent;
            y->link[y->degree] = x->l2par;
            y->child[y->degree++] = x;
            test.tail = x;
        }

    conductors = 0;
    x = test.head;
    do
        if (x->degree) {
            if (x->through != 2) x->through = cond_best[conductors++];
        } else x->through = 0;
        while (x = x->next_n);

    if (test.head != test.tail) {
        if (ImageOrQuery) test.SolveImage(L, ab, cd, 1);
        else test.SolveQuery(L, lab, lcd, 1);
    }
}
//**************************************************************

void Network::OptimumQuery(Network &test, long L, double b, double d) {
    this->AuxOptimum(0, test, L, (double) b, (double) d);
}
//**************************************************************

void Network::OptimumImage(Network &test, long L, double a, double c) {
    this->AuxOptimum(1, test, L, a, c);
}
//**************************************************************

void Network::GreedyQuery(long start_L, long end_L, double b, double d) {
    this->AuxGreedy(0, start_L, end_L, (double) b, (double) d);
}
//**************************************************************

void Network::GreedyImage(long start_L, long end_L, double a, double c) {
    this->AuxGreedy(1, start_L, end_L, a, c);
}
//**************************************************************

void Network::GreedyQuery(Network &test, long L, double b, double d) {
    this->Aux2Greedy(0, test, L, (double) b, (double) d);
}
//**************************************************************

void Network::GreedyImage(Network &test, long L, double a, double c) {
    this->Aux2Greedy(1, test, L, a, c);
}
//**************************************************************

void Network::PrintNet(void) {
    Node *x;
    int i;

    x = head;
    while (x) {
        if (x->parent)
            printf("%s (%f %f %f %i %i %f %f) with parent %s", x->name, x->power, x->e0, x->l2par, x->fe, x->through, x->aggregate_p, x->aggregate_e, x->parent->name);
        else
            printf("%s (%f %f %f %i %i %f %f) ", x->name, x->power, x->e0, x->l2par, x->fe, x->through, x->aggregate_p, x->aggregate_e);
        if (x->degree) {
            printf(". Child nodes : ");
            for (i = 0; i < x->degree; i++) printf("%s ", x->child[i]->name);
        }
        printf("\n");
        x = x->next_n;
    }
}
//**************************************************************

double Network::Quantify(bool ImageOrQuery, double ab, double cd) {
    static Network net1, net2;
    double t1, t2;

    net1 = *this;
    net2 = *this;

    net1.AuxQuantify1();
    net2.AuxQuantify2();
    if (ImageOrQuery) {
        t1 = net1.SimulateImage(ab, cd, 0);
        t2 = net2.SimulateImage(ab, cd, 0);
    } else {
        t1 = net1.SimulateQuery((long) ab, (long) cd, 0);
        t2 = net2.SimulateQuery((long) ab, (long) cd, 0);
    }

    if (t1 < t2) // Quantify1 usually gets the furhest of nodes
    {
        (*this).AuxQuantify1();
        (*this).best_quant = 1;
        (*this).quant_per_better = (t2 - t1) / t2;
        return (t1);
    } else {
        (*this).AuxQuantify2();
        (*this).best_quant = 2;
        (*this).quant_per_better = (t1 - t2) / t1;
        return (t2);
    }
}
//**************************************************************
// Faster nodes are favored for rounding up. After rounding
// there are 3 cases: (1) the original load is maintained
// (2) the load is underestimated and (3) the load is
// overestimated.
// It can be called only after a solution is found.

void Network::AuxQuantify1() {
    Queue temp_queue;
    Queue load_update;
    long temp_L;
    int additions;
    Node *x, *y;
    double aux, aux2, max_pow;
    double &min_pow = max_pow;
    double scale;
    int i, j, k;
    int in_array;
    int mark;

    sum_dL = 0;
    max_dL = 0;
    min_dL = 0;
    num_elem_dL = 0;
    head->Lint = (long) head->L;
    if (head != tail) {
        temp_queue.Insert(head);
        while (!temp_queue.IsEmpty()) {
            x = temp_queue.Pop();
            if (!x->Lint) {
                x->part = 0;
                for (i = 0; i < x->degree; i++) {
                    x->child[i]->Lint = 0;
                    temp_queue.Insert(x->child[i]);
                }
            } else {
                x->visited = 0;
                aux = x->Lint * x->part;
                temp_L = (long int) floor(aux);
                x->time = (x->part * x->L - ceil(aux)) / (x->part * x->L);
                for (i = 0; i < x->degree; i++) {
                    y = x->child[i];
                    if (y->degree) {
                        aux = y->L / x->L; // y->part for subtree 
                        aux *= x->Lint;
                        temp_L += (long int) floor(aux);
                        y->Lint = (long int) floor(aux);
                        y->time = (y->L - ceil(aux)) / (y->L);
                        temp_queue.Insert(x->child[i]);
                    } else {
                        aux = y->part * x->Lint;
                        temp_L += (long int) floor(aux);
                        y->Lint = (long int) floor(aux);
                        y->time = (y->part * x->L - ceil(aux)) / (y->part * x->L);
                    }
                }

                additions = x->Lint - temp_L;
                if (additions > 0) //---->(2) the load is underestimated
                {
                    in_array = 0;
                    if (!x->through)
                        nodes_array[in_array++] = x;

                    for (i = 0; i < x->degree; i++)
                        nodes_array[in_array++] = x->child[i];

                    // now sort the examined nodes by decreasing load difference
                    for (j = 0; j < in_array; j++) {
                        mark = j;
                        max_pow = nodes_array[j]->time;
                        for (k = j + 1; k < in_array; k++) {
                            aux = nodes_array[k]->time;
                            if (max_pow < aux) {
                                max_pow = aux;
                                mark = k;
                            }
                        }
                        if (mark != j) // switch pointers
                        {
                            y = nodes_array[j];
                            nodes_array[j] = nodes_array[mark];
                            nodes_array[mark] = y;
                        }
                    }


                    for (i = 0; i < additions; i++) {
                        y = nodes_array[i % in_array];
                        if (y == x) {
                            x->visited = 1;
                            aux = x->Lint * x->part;

                            aux2 = x->part * x->L;
                            x->part = ceil(aux) / x->Lint;

                            aux2 = ceil(aux) - aux2;
                            max_dL = (max_dL < aux2) ? aux2 : max_dL;
                            min_dL = (min_dL > aux2) ? aux2 : min_dL;
                            sum_dL += fabs(aux2);
                            num_elem_dL++;
                        } else y->Lint++;
                    }
                }


                // gather statistics for each leaf node
                for (i = 0; i < x->degree; i++) {
                    y = x->child[i];
                    if (!y->degree) {
                        aux2 = y->Lint - y->part * x->L;
                        max_dL = (max_dL < aux2) ? aux2 : max_dL;
                        min_dL = (min_dL > aux2) ? aux2 : min_dL;
                        sum_dL += fabs(aux2);
                        num_elem_dL++;
                    }
                }


                // check if the x->part has been modified 
                if (!x->visited && !x->through) {
                    aux2 = x->Lint * x->part;
                    aux = floor(aux2);
                    x->part = aux / (x->Lint);

                    aux2 = aux - aux2;
                    max_dL = (max_dL < aux2) ? aux2 : max_dL;
                    min_dL = (min_dL > aux2) ? aux2 : min_dL;
                    sum_dL += fabs(aux2);
                    num_elem_dL++;
                }
            }
        }
    }

    // prepare L fields for simulation routines
    x = head;
    while (x) {
        x->L = x->Lint;
        if (!x->degree)
            if (x->parent->Lint)
                x->part = (1.0 * x->Lint) / x->parent->Lint;
            else
                x->part = 0;
        x = x->next_n;
    }

    this->ClipIdleNodes();
}

//**************************************************************
// Clipping done on the basis of Lint assigned to the node

void Network::ClipIdleNodes() {
    Node *x, *y;
    int i;

    // Remove idle nodes
    if (clipping == 0) // no node is to be removed
    {
        x = head->next_n; // HOST should not be removed
        while (x) {
            if (!x->Lint) {
                clipping = 1;
                x = NULL;
                continue;
            }
            x = x->next_n;
        }
    } else {
        y = head->next_n; // HOST should not be removed
        x = y->next_n;
        while (x) {
            if (!x->Lint) {
                clipping = 1;
                y->next_n = x->next_n;

                x->next_n = redundant; // keep list of redundant nodes
                redundant = x;

                i = 0;
                while (x->parent->child[i] != x) i++;
                for (; i < x->parent->degree - 1; i++) {
                    x->parent->link[i] = x->parent->link[i + 1];
                    x->parent->child[i] = x->parent->child[i + 1];
                }
                x->parent->degree--;
                if (!x->parent->degree) {
                    if (x->parent != head)
                        x->parent->part = (1.0 * x->parent->Lint) / x->parent->parent->Lint;
                }
                x = y->next_n; // y is not changed
            } else {
                y = x;
                x = x->next_n;
            }
        }
    }
}
//**************************************************************

void Network::AuxQuantify2() {
    Queue temp_queue;
    Queue load_update;
    long temp_L;
    int additions;
    Node *x, *y;
    double aux, aux2, max_pow;
    double &min_pow = max_pow;
    double scale;
    int i, j, k;
    Node * nodes_array[MAX_NODE_DEGREE + 1]; // the root node is also inserted
    int in_array;
    int mark;

    sum_dL = 0;
    max_dL = 0;
    min_dL = 0;
    num_elem_dL = 0;
    head->Lint = (long int) head->L;
    if (head != tail) {
        temp_queue.Insert(head);
        while (!temp_queue.IsEmpty()) {
            x = temp_queue.Pop();
            if (!x->Lint) {
                x->part = 0;
                for (i = 0; i < x->degree; i++) {
                    x->child[i]->Lint = 0;
                    temp_queue.Insert(x->child[i]);
                }
            } else {
                x->visited = 0;
                aux = x->Lint * x->part;
                temp_L = (long int) floor(aux);
                x->time = (x->part * x->L - floor(aux)) / (x->part * x->L);
                for (i = 0; i < x->degree; i++) {
                    y = x->child[i];
                    if (y->degree) {
                        aux = y->L / x->L; // y->part for subtree 
                        aux *= x->Lint;
                        temp_L += (long int) floor(aux);
                        y->Lint = (long int) floor(aux);
                        y->time = (y->L - floor(aux)) / (y->L);
                        temp_queue.Insert(x->child[i]);
                    } else {
                        aux = y->part * x->Lint;
                        temp_L += (long int) floor(aux);
                        y->Lint = (long int) floor(aux);
                        y->time = (y->part * x->L - floor(aux)) / (y->part * x->L);
                    }
                }

                additions = x->Lint - temp_L;
                if (additions > 0) //---->(2) the load is underestimated
                {
                    in_array = 0;
                    if (!x->through)
                        nodes_array[in_array++] = x;

                    for (i = 0; i < x->degree; i++)
                        nodes_array[in_array++] = x->child[i];

                    // now sort the examined nodes by decreasing load difference
                    for (j = 0; j < in_array; j++) {
                        mark = j;
                        max_pow = nodes_array[j]->time;
                        for (k = j + 1; k < in_array; k++) {
                            aux = nodes_array[k]->time;
                            if (max_pow < aux) {
                                max_pow = aux;
                                mark = k;
                            }
                        }
                        if (mark != j) // switch pointers
                        {
                            y = nodes_array[j];
                            nodes_array[j] = nodes_array[mark];
                            nodes_array[mark] = y;
                        }
                    }


                    for (i = 0; i < additions; i++) {
                        y = nodes_array[i % in_array];
                        if (y == x) {
                            x->visited = 1;
                            aux = x->Lint * x->part;

                            aux2 = x->part * x->L;
                            x->part = ceil(aux) / x->Lint;

                            aux2 = ceil(aux) - aux2;
                            max_dL = (max_dL < aux2) ? aux2 : max_dL;
                            min_dL = (min_dL > aux2) ? aux2 : min_dL;
                            sum_dL += fabs(aux2);
                            num_elem_dL++;
                        } else y->Lint++;
                    }
                }


                // gather statistics for each leaf node
                for (i = 0; i < x->degree; i++) {
                    y = x->child[i];
                    if (!y->degree) {
                        aux2 = y->Lint - y->part * x->L;
                        max_dL = (max_dL < aux2) ? aux2 : max_dL;
                        min_dL = (min_dL > aux2) ? aux2 : min_dL;
                        sum_dL += fabs(aux2);
                        num_elem_dL++;
                    }
                }


                // check if the x->part has been modified 
                if (!x->visited && !x->through) {
                    aux2 = x->Lint * x->part;
                    aux = floor(aux2);
                    x->part = aux / (x->Lint);

                    aux2 = aux - aux2;
                    max_dL = (max_dL < aux2) ? aux2 : max_dL;
                    min_dL = (min_dL > aux2) ? aux2 : min_dL;
                    sum_dL += fabs(aux2);
                    num_elem_dL++;
                }
            }
        }
    }

    // prepare L fields for simulation routines
    x = head;
    while (x) {
        x->L = x->Lint;
        if (!x->degree)
            if (x->parent->Lint)
                x->part = (1.0 * x->Lint) / x->parent->Lint;
            else
                x->part = 0;
        x = x->next_n;
    }


    this->ClipIdleNodes();
}
//**************************************************************

/* Calculates the best transition between optimum distributions
 for different loads. It works only for query processing
 since in image processing the data are transmitted every
 time from the host processor.
 ----> This procedure works only if load1 <= load2. Cases
 where load2 is produced by dropping a part of load1 must
 be preprocessed in the following way:
 - drop from every node the part that is to be dropped
 - then load1' ==load2 and so this procedure can be applied.
 */
void Network::TransQueryDistr(Network &distr1, Network &distr2) {
    Queue temp_queue;
    Queue temp2_queue;
    double aux;
    Node *p, *q;
    int i, j;

    p = head;
    while (p) {
        if (p->degree) p->visited = p->degree;
        p->L = p->part = 0; // The part field is used to  hold
        // the locally redistributed load
        p = p->next_n;
    }
    q = distr1.head;
    while (q) {
        p = head;
        while (p->ID != q->ID) p = p->next_n;
        if (!q->degree && q->parent) aux = q->part * q->parent->L;
        else aux = q->L;
        p->L = aux;
        q = q->next_n;
    }
    q = distr2.head;
    while (q) {
        p = head;
        while (p->ID != q->ID) p = p->next_n;
        if (!q->degree && q->parent) aux = q->part * q->parent->L;
        else aux = q->L;
        p->L -= aux;
        q = q->next_n;
    }

    p = head;
    while (p) {
        printf("%s %f\n", p->name, p->L);
        p = p->next_n;
    }

    p = head;
    while (p) {
        if (p->parent && !p->degree)
            if (!--p->parent->visited) temp_queue.Insert(p->parent);
        p = p->next_n;
    }

    // packets travel to host
    while (!temp_queue.IsEmpty()) {
        p = temp_queue.Pop();

        //unload sources
        for (i = 0; i < p->degree; i++)
            if (p->child[i]->L > 0) {
                printf("%s sends %f to %s\n", p->child[i]->name, p->child[i]->L, p->name);
                p->part += p->child[i]->L; // to be distributed locally
                p->child[i]->L = 0;
            }

        if (p->parent) // not for host node
        {
            //distribute locally
            for (i = 0; (i < p->degree) && p->part; i++)
                if (p->child[i]->L < 0) {
                    aux = p->child[i]->part = MIN(-p->child[i]->L, p->part);
                    p->part -= aux;
                    printf("%s sends %.0f to %s\n", p->name, aux, p->child[i]->name);
                    temp2_queue.Insert(p->child[i]);
                }
        } else if (!p->parent && p->L) // only for host node
        {
            p->part += -p->L;
            temp2_queue.Insert(p->child[0]->parent);
        }

        // packets travel to leaves
        while (!temp2_queue.IsEmpty()) {
            q = temp2_queue.Pop();
            q->L += q->part;
            if (q->degree) {
                for (i = 0; (i < q->degree) && q->part; i++)
                    if (q->child[i]->L < 0) {
                        aux = q->child[i]->part = MIN(-q->child[i]->L, q->part);
                        q->part -= aux;
                        printf("%s sends %.0f to %s\n", q->name, aux, q->child[i]->name);
                        temp2_queue.Insert(q->child[i]);
                    }
            }
            q->part = 0;
        }

        if (p->parent)
            if (!--p->parent->visited) temp_queue.Insert(p->parent);
    }
}
//**************************************************************
// Query=1. The parameter L is for the generation of e(i)'s
// if all_fe==0 then FE is random. If -1 NFE, if 1 FE

void Network::GenerateRandomTree(bool ImageOrQuery, int N,
        float min_p, float max_p,
        float min_l, float max_l,
        float min_e, float max_e, bool full_tree = false, bool all_fe = false) {
    int i, fe;
    float l, p, e;
    int parent;
    Node *x;
    char temp[5];

    l = (max_l - min_l) * ran2(&global_random_seed) + min_l;

    if (all_fe == 1) fe = 1;
    else if (all_fe == -1) fe = 0;
    else fe = (int) (2 * ran2(&global_random_seed));
    p = (max_p - min_p) * ran2(&global_random_seed) + min_p;
    e = (max_e - min_e) * ran2(&global_random_seed) + min_e;
    this->InsertNode((char*) "HOST", p, e, (char*) "", l, fe);
    for (i = 0; i < N - 1; i++) {
        if (!full_tree) // choose a random parent or fill in series the tree to itsfull capacity
        {
            parent = (int) round(i * 1.0 * ran2(&global_random_seed));
            x = this->head;
            while (parent--) {
                x = x->next_n;
            }
            if (x->degree == sched_lib_max_node_degree) {
                x = this->head;
                while (x->degree == sched_lib_max_node_degree) {
                    x = x->next_n;
                }
            }
        } else // fill tree
        {
            x = this->head;
            while (x->degree == sched_lib_max_node_degree) {
                x = x->next_n;
            }
        }

        if (all_fe == 1) fe = 1;
        else if (all_fe == -1) fe = 0;
        else fe = (int) (2 * ran2(&global_random_seed));
        p = (max_p - min_p) * ran2(&global_random_seed) + min_p;
        e = (max_e - min_e) * ran2(&global_random_seed) + min_e;
        if (ImageOrQuery)
            l = (max_l - min_l) * ran2(&global_random_seed) + min_l;
        sprintf(temp, "%i", i);
        this->InsertNode(temp, p, e, x->name, l, fe);
    }
    this->head->power = this->head->next_n->power + 1;
}
//**************************************************************
// Query=1.

void Network::ReUseRandomTree(bool ImageOrQuery,
        float min_p, float max_p,
        float min_l, float max_l,
        float min_e, float max_e, bool full_tree = false, bool all_fe = false) {
    int i;
    float l;
    int parent;
    Node *x, *y, *z;

    if (this->redundant) // get the nodes that were forced out of the tree
    {
        x = this->head;
        while (x->next_n) x = x->next_n;

        while (this->redundant) {
            x->next_n = this->redundant;
            this->redundant = this->redundant->next_n;
        }
        this->tail = x;
    }

    l = (max_l - min_l) * ran2(&global_random_seed) + min_l;
    x = this->head;
    while (x) {
        if (all_fe == 1) x->fe = 1;
        else if (all_fe == -1) x->fe = 0;
        else x->fe = (int) (2 * ran2(&global_random_seed));

        x->power = (max_p - min_p) * ran2(&global_random_seed) + min_p;
        x->e0 = (max_e - min_e) * ran2(&global_random_seed) + min_e;
        x->degree = 0;
        if (ImageOrQuery)
            l = (max_l - min_l) * ran2(&global_random_seed) + min_l;
        x->l2par = l;
        x = x->next_n;
    }

    x = this->head->next_n;
    i = 0;
    while (x) {
        if (!full_tree) {
            parent = (int) round(i * ran2(&global_random_seed));
            y = this->head;
            while (--parent >= 0) {
                y = y->next_n;
            }
            if (y->degree == sched_lib_max_node_degree) {
                y = this->head;
                while (y->degree == sched_lib_max_node_degree) {
                    y = y->next_n;
                }
            }
        } else {
            y = this->head;
            while (y->degree == sched_lib_max_node_degree) y = y->next_n;
        }


        y->child[y->degree] = x;
        x->parent = y;
        y->link[y->degree++] = x->l2par;
        x = x->next_n;
        i++;
    }

}

//**************************************************************
void Network::UniformQueryGreedy(Network &test, long L, double b, double d) {
  this->UniformGreedy(0, test, L, b, d);
}
//**************************************************************
void Network::UniformImageGreedy(Network &test, long L, double a, double c) {
  this->UniformGreedy(1, test, L, a, c);
}

//**************************************************************
// A different approach is needed for a uniform tree

void Network::UniformGreedy(bool ImageOrQuery, Network &test, long L, double ab, double cd) {
    Queue temp_queue;
    Node *x, *y; // holds the newly inserted node
    unsigned int in_array = 0, num_nodes;
    int i, j, k;
    double min_time, temp_time, local_min;
    int best_i; // index of best newly inserted node. -1 for no insertion
    long lab = (long) ab, lcd = (long) cd;
    Node *first = NULL;
    int path_len, best_path_len;

    test = *this;

    num_nodes = strlen(test.node_usage);
    in_array = test.netnode[0].degree;
    for (i = 0; i < in_array; i++)
        nodes_array[i] = test.netnode[0].child[i];

    // Fake the removal of all nodes except host node
    for (i = 0; i < num_nodes; i++) {
        test.netnode[i].degree = 0;
        test.netnode[i].next_n = NULL;
    }
    test.tail = test.head;


    if (head->through) min_time = FLT_MAX;
    else min_time = test.head->power * (L + test.head->e0);
    do {
        //printf("\n");
        best_i = -1;
        best_path_len = 0;
        for (i = 0; i < in_array; i++) {
            test.InsertDuplicateNode(nodes_array[i]);
            if (ImageOrQuery) test.SolveImage(L, ab, cd);
            else test.SolveQuery(L, lab, lcd);
            if (test.valid == 1) {
                temp_time = test.head->aggregate_p * (L + test.head->aggregate_e);
                if (temp_time < min_time) {
                    min_time = temp_time;
                    best_i = i;
                    best_path_len = 0;
                    x = nodes_array[i];
                    while (x) {
                        x = x->parent;
                        best_path_len++;
                    }
                }
            }
            test.RemoveDuplicateNode();
        }
        if (best_i >= 0) {
            if (!first) first = nodes_array[best_i];
            test.InsertDuplicateNode(nodes_array[best_i]);
            x = test.head;
            while (x->next_n) x = x->next_n;
            test.tail = x;

            // insert its child nodes in nodes_array
            i = 1;
            do {
                if (test.netnode[i].parent == nodes_array[best_i]) {
                    nodes_array[in_array++] = &test.netnode[i];
                }
                i++;
            } while (i < num_nodes);
            // remove newly inserted node from nodes_array
            i = 0;
            do {
                if (nodes_array[i] != x) i++;
                else {
                    nodes_array[i] = nodes_array[--in_array];
                    i = in_array;
                }
            } while (i < in_array);
        }
    } while (in_array && (best_i >= 0)); // if any nodes left for addition

    if (test.head != test.tail)
        if (ImageOrQuery) test.SolveImage(L, ab, cd);
        else test.SolveQuery(L, lab, lcd);
}

//**************************************************************
// sort the nodes for an image-query operation
// homo=true for a homogeneous

void ImageQuerySort(Node *temp, bool homo, int N = 0) {
    Node *temp2;
    int i, j;
    double aux, min_lp;
    int mark;

    if (N == 0) N = temp->degree;

    // bubblesort
    for (i = 0; i < N - 1; i++) {
        mark = i;
        min_lp = homo ? -temp->child[i]->e0 : temp->child[i]->power;
        for (j = i + 1; j < N; j++) {
            aux = homo ? -temp->child[j]->e0 : temp->child[j]->power;
            if (min_lp > aux) {
                min_lp = aux;
                mark = j;
            }
        }
        if (mark != i) // switch pointers
        {
            temp2 = temp->child[i];
            aux = temp->link[i];
            temp->child[i] = temp->child[mark];
            temp->link[i] = temp->link[mark];
            temp->child[mark] = temp2;
            temp->link[mark] = aux;
        }
    }
}
//---------------------------------------------
// Assumes that the network contains N+1 nodes in a single level tree
// with the children doing all the work. The master is supposed to be the
// load originating node
// L is a reference because it can be modified

double Network::SolveImageQueryHomogeneous(long &L, long b, double d, bool firstcall = false) {

    Node *temp;
    int i, k, M, N = head->degree;
    double p, l, t, aux;
    long added = 0;

    if (L == 0 && b == d) // check the case where the equal partitioning is the optimum one
        return EqualDistrImageQuery(L, b, d);

    if (L == 0) added = L = 1;

    temp = head;
    p = temp->child[0]->power;
    l = temp->child[0]->l2par;

    for (i = 0; i < N; i++) // reset through flag. Also done before returning
        temp->child[i]->through = 0;

    t = SolveImageQueryHomo_Aux(L, b, d);
    if (valid == 0 && firstcall) // in this case, a subset of nodes should be used
        return -1;

    if (valid == 0)
        for (i = 0; i < N; i++)
            temp->child[i]->time = temp->child[i]->e0; // temporarily store previous e0

    long reserv = 0;
    long laux;

    long iter = 0;
    int last_removed = N - 1;
    while (valid == 0 || reserv != 0 || added > 0) {
        reserv = 0;
        for (i = 0; i < N; i++)
            if (temp->child[i]->part < 0) {
                laux = (long) floor(temp->child[i]->e0 + temp->child[i]->part * L);
                if (laux < 0) {
                    valid = 0; // a subset of nodes should be used
                    return -1;
                } else {
                    int aux = (long) ceil(-temp->child[i]->part * L);
                    reserv += aux;
                    temp->child[i]->e0 -= aux;
                }
            } else if (temp->child[i]->time > temp->child[i]->e0) {
                double aux = temp->child[i]->part * L + temp->child[i]->e0;
                long diff;
                if (aux > temp->child[i]->time)
                    diff = (long) floor(temp->child[i]->time - temp->child[i]->e0);
                else
                    diff = (long) floor(aux - temp->child[i]->e0);
                reserv -= diff;
                temp->child[i]->e0 += diff;
            }

        L += reserv;

        if (added > 0) {
            L -= added; // removes the small load that was added in order to get a first solution
            added = 0;
            reserv = 1;
        }

        // examine the extreme case where although part_i <0, L stays 0
        if (L == 0) {
            for (int i = 0; i < N; i++)
                temp->child[i]->part = 0;
            return SimulateImageQuery(L, b, d);
        }
        printf("%i Aux %li: %li, %lf %li\n", N, iter, L, t, reserv);

        iter++;
        if (iter > ALG1_MAX_ITER)
        {
            if (last_removed < 1) // error checking
            {
                printf("Error in last_removed\n");
                for (int n = 0; n < N; n++)
                    printf("%s : %lf  %lf %lf\n", temp->child[n]->name, temp->child[n]->part, temp->child[n]->e0, temp->child[n]->time);
                exit(2);
            }
            temp->child[last_removed]->part = 0;
            temp->child[last_removed]->through = 3;
            last_removed--;
        }
        t = SolveImageQueryHomo_Aux(L, b, d); // not sorting the nodes gives inferior results 
    }

    // reset through flag
    for (i = 0; i < N; i++)
        temp->child[i]->through = 0;
    valid = 1;
    if (iter > ALG1_MAX_ITER) return SimulateImageQuery(L, b, d); // if the nodes have been "truncated"
    return t;
}
//---------------------------------------------
// This is a building block for the SolveImageQueryHomo function
// Solves the homogeneous system case of image query
// Assumes that the network contains N+1 nodes in a single level tree
// with the children doing all the work. The master is supposed to be the
// load originating node
// In the following function, L cannot be 0

double Network::SolveImageQueryHomo_Aux(long L, long b, double d, bool sortflag) {
    Node *temp;
    int i, k, N = head->degree;
    double p, l, aux, sum = 0, part0;
    double ppl, ppl_pow[N];
    double sum_eeL[N];


    temp = head;
    head->L = L;
    p = temp->child[0]->power;
    l = temp->child[0]->l2par;

    for (i = 1; i < N; i++)
        if (temp->child[i]->through == 3) {
            N = i;
            break;
        }
    // the sorting step has to be called after the number of nodes getting a piece of L is found
    if (sortflag)
        ImageQuerySort(temp, true, N);

    ppl = p / (p + l);
    ppl_pow[0] = 1;
    for (i = 1; i < N; i++)
        ppl_pow[i] = ppl_pow[i - 1] * ppl;

    //calculate part_0
    for (i = 1; i < N; i++) {
        sum_eeL[i] = 0;
        for (k = 0; k < i; k++) {
            aux = (temp->child[k]->e0 - temp->child[k + 1]->e0) / L;
            aux = aux * ppl_pow[i - k];
            sum_eeL[i] += aux;
        }
        sum += sum_eeL[i];
    }
    part0 = (1.0 * d - b) / L + (l + l * N * (b - d) / L - l * sum) / (p + l - p * ppl_pow[N - 1]);
    temp->child[0]->part = part0;

    part0 = part0 + (1.0 * b - d) / L;
    for (i = 1; i < N; i++)
        temp->child[i]->part = part0 * ppl_pow[i] - (1.0 * b - d) / L + sum_eeL[i];

    valid = 1;
    aux = 0;
    for (i = 0; i < N; i++) {
        aux += temp->child[i]->part;
    }

    if (fabs(aux - 1) > 0.0001) valid = 0;

    for (i = 0; i < N; i++)
        if (temp->child[i]->part < 0) {
            valid = 0;
            break;
        }

    //return total execution time
    part0 = temp->child[0]->part;
    aux = l * (part0 * L + b) + p * (part0 * L + temp->child[0]->e0) + N * l*d;
    return aux;
}
//---------------------------------------------------
// Used to estimate the running time for any partitioning. 1-port only

double Network::SimulateImageQuery(long L, long b, double d) {
    Node *temp;
    int i, N = head->degree;
    double l, p, e, part, t_distr, t_coll, t;

    temp = head;
    t_distr = 0;
    t_coll = 0;

    for (i = 0; i < N; i++) {
        l = temp->child[i]->l2par;
        p = temp->child[i]->power;
        e = temp->child[i]->e0;
        part = temp->child[i]->part;

        t_distr += l * (L * part + b);
        t = t_distr + p * (L * part + e) + l*d; // assumes that there is no conflict in the result collections
        t_coll = (t >= t_coll) ? t : t_coll;
    }
    return t_coll;
}
//---------------------------------------------------
// Used to estimate the running time for any partitioning. 1-port only stream-type tasks

double Network::SimulateImageQuery_ST(long L, long b, double d) {
    Node *temp;
    int i, N = head->degree;
    double l, p, e, part, t_distr, t_coll, t;

    temp = head;
    t_distr = 0;
    t_coll = 0;

    for (i = 0; i < N; i++) {
        l = temp->child[i]->l2par;
        p = temp->child[i]->power;
        e = temp->child[i]->e0;
        part = temp->child[i]->part;
        if (l < p) // is it computationally bound?
        {
            t_distr += l*b;
            t = t_distr + p * (L * part + e) + l*d; // assumes that there is no conflict in the result collections
            t_distr += l * L*part;
            t_coll = (t >= t_coll) ? t : t_coll;
        } else {
            t_distr += l*b;
            if (p * (L * part + e - b) > l * L * part)
                t = t_distr + p * (L * part + e) + l * d; // assumes that there is no conflict in the result collections
            else
                t = t_distr + l * L * part + p * b + l*d; // assume that b=I
            t_distr += l * L*part;
            t_coll = (t >= t_coll) ? t : t_coll;
        }
    }
    return t_coll;
}
//---------------------------------------------
// Used to make the last load assigned to a node, part of its cache
// Actually works for any platform and not only homogeneous

void Network::ImageQueryHomoEmbed(long L) {
    ImageQueryEmbed(L);
}
//---------------------------------------------
// Used to make the last load assigned to a node, part of its cache
// Actually works for any platform and not only homogeneous

void Network::ImageQueryEmbed(long L) {
    Node *temp;
    int i, N = head->degree;

    temp = head;
    head->L = 0;
    for (i = 0; i < N; i++)
        temp->child[i]->e0 += temp->child[i]->part * L;
}
//---------------------------------------------
// Used to make the last load assigned to a node, part of its cache
// Actually works for any platform and not only homogeneous

void Network::ImageQueryEmbed_Ninst(long L, int M) {
    Node *temp;
    int i, j, N = head->degree;

    temp = head;
    head->L = 0;
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++)
            temp->child[i]->e0 += temp->child[i]->mi_part[j] * L;
}
//---------------------------------------------
// Used for comparing against the non-uniform distribution
// Load is transferred in case nodes are added or removed
// L is a reference because it can be modified

double Network::EqualDistrImageQuery(long &L, long b, double d) {
    Node *temp;
    int i, N = head->degree;
    double part, aux, Lsum;
    long L1;


    temp = head;
    head->L = L;
    Lsum = L;
    for (i = 0; i < N; i++)
        Lsum += temp->child[i]->e0;

    part = Lsum / N;
    aux = 0;
    for (i = 0; i < N; i++)
        if (temp->child[i]->e0 > part) {
            aux += temp->child[i]->e0 - part; // this needs to be communicated to other nodes
            temp->child[i]->e0 = part;
            temp->child[i]->part = 0;
        }

    L += (long) ceil(aux); // this needs to be communicated  

    for (i = 0; i < N; i++)
        if (temp->child[i]->e0 < part) {
            aux = part - temp->child[i]->e0;
            aux /= L;
            temp->child[i]->part = aux;
        }
    return SimulateImageQuery(L, b, d);
}
//---------------------------------------------
// Derived from EqualDistrImageQuery for a heterogeneous platform

double Network::EqualDistrImageQueryHeterog(long &L, long b, double d) {
    Node *temp;
    int i, N = head->degree;
    double part, aux, Lsum;
    long L1;
    double denom = 0;

    temp = head;
    head->L = L;

    for (i = 0; i < N; i++)
        denom += (1.0 / temp->child[i]->power);

    aux = 0;
    for (i = 0; i < N; i++)
        temp->child[i]->part = 1.0 / temp->child[i]->power / denom;

    valid = 1;

    aux = 0;
    for (i = 0; i < N; i++)
        aux += temp->child[i]->part;

    if (fabs(aux - 1) > 0.0001) {
        printf("EqualDistrImageQueryHeterog invalid because total sum is %lf\n", aux);
        valid = 0;
    }

    for (i = 0; i < N; i++)
        if (temp->child[i]->part < 0) {
            valid = 0;
            break;
        }

    return SimulateImageQuery(L, b, d);
}
//---------------------------------------------

double Network::SolveImageQuery_NPort_Aux(long L, long b, double d) {
    Node *temp;
    int i, k, M, N = head->degree;
    double t, part0, aux, p0e0, denom, nomin;
    double p[N], e[N], l[N];

    temp = head;
    head->L = L;

    if (N == 1) {
        temp->child[0]->part = 1;
        return temp->child[0]->l2par * (L + b) + temp->child[0]->power * (L + temp->child[0]->e0) + temp->child[0]->l2par * d;
    }

    for (i = 0; i < N; i++) {
        p[i] = temp->child[i]->power;
        e[i] = temp->child[i]->e0;
        l[i] = temp->child[i]->l2par;
    }

    //calculate part_0
    p0e0 = p[0] * e[0];
    nomin = 0;
    denom = 0;
    for (i = 1; i < N; i++)
        denom += 1.0 / (p[i] + l[i]);
    denom *= (p[0] + l[0]);
    denom++;

    for (i = 1; i < N; i++)
        nomin += ((p[i] * e[i] - p0e0 + (b + d)*(l[i] - l[0])) / (p[i] + l[i]));
    nomin /= L;
    nomin++;

    part0 = nomin / denom;
    temp->child[0]->part = part0;

    // compute remaining parts
    for (i = 1; i < N; i++)
        temp->child[i]->part = part0 * (p[0] + l[0]) / (p[i] + l[i]) + (p0e0 - p[i] * e[i] + (b + d)*(l[0] - l[i])) / (L * (p[i] + l[i]));

    // check solution validity
    valid = 1;
    aux = 0;
    for (i = 0; i < N; i++)
        aux += temp->child[i]->part;

    if (fabs(aux - 1) > 0.0001) {
        printf("Nport invalid because total sum is %lf\n", aux);
        valid = 0;
    }

    for (i = 0; i < N; i++)
        if (temp->child[i]->part < 0) {
            valid = 0;
            break;
        }

    //return total execution time
    aux = l[0] * (part0 * L + b) + p[0] * (part0 * L + e[0]) + l[0] * d;
    return aux;
}
//---------------------------------------------------
// Used to estimate the running time for any partitioning. N-port only

double Network::SimulateImageQuery_Nport(long L, long b, double d, bool isstream) {
    Node *temp;
    int i, N = head->degree;
    double t, aux, I = b;
    double p[N], e[N], l[N];

    temp = head;
    for (i = 0; i < N; i++) {
        p[i] = temp->child[i]->power;
        e[i] = temp->child[i]->e0;
        l[i] = temp->child[i]->l2par;
    }
    if (!isstream) // block-type tasks
    {
        t = 0;
        for (i = 0; i < N; i++) {
            aux = l[i] * (temp->child[i]->part * L + b + d) + p[i] * (temp->child[i]->part * L + e[i]);
            if (aux > t) t = aux;
        }
    } else {
        double t_commb, t_compb; // comm. and comp. bound
        t = 0;

        for (i = 0; i < N; i++) {
            if (e[i] == 0)
                t_compb = l[i]*(b + I + d) + p[i] * temp->child[i]->part * L; // I=b
            else
                t_compb = l[i]*(b + d) + p[i]*(temp->child[i]->part * L + e[i]);
            t_commb = l[i]*(b + d + temp->child[i]->part * L) + p[i] * I;
            aux = (t_commb > t_compb) ? t_commb : t_compb;
            if (aux > t) t = aux;
        }
    }
    return t;
}
//---------------------------------------------------
// Applies Algorithm 1 for managing the cache. Works for heterogeneous platform

double Network::SolveImageQuery_NPort(long &L, long b, double d) {
    Node *temp;
    int i, k, M, N = head->degree;
    double p, l, t, aux;
    long added = 0;

    if (L == 0) added = L = 1;

    temp = head;
    head->L = L;
    p = temp->child[0]->power;
    l = temp->child[0]->l2par;

    t = SolveImageQuery_NPort_Aux(L, b, d);

    if (valid == 1 && added == 0) return t;

    // Solution is invalid, examine L redistribution 
    for (i = 0; i < N; i++)
        temp->child[i]->time = temp->child[i]->e0; // temporarily store previous e0

    double reserv = 0;
    long laux;

    long iter = 0;
    while (valid == 0 || reserv != 0 || added != 0) // cause the added load to be fixed
    {
        reserv = 0;
        for (i = 0; i < N; i++)
            if (temp->child[i]->part < 0) {
                laux = (long) floor(temp->child[i]->e0 + temp->child[i]->part * L);
                if (laux < 0) {
                    valid = 0; // a subset of nodes should be used
                    return -1;
                } else {
                    reserv += (temp->child[i]->e0 - laux);
                    temp->child[i]->e0 = laux;
                }
            }

        double missing = (long) ceil(reserv) - reserv;
        if (missing > 0) {
            for (i = 0; i < N; i++)
                if (temp->child[i]->e0 > missing) {
                    temp->child[i]->e0 -= missing;
                    break;
                }
        }
        L += (long) ceil(reserv);

        if (added > 0) {
            L -= added; // removes the small load that was added in order to get a first solution
            added = 0;
            reserv = 1; // just a token load to force recalculation
        }

        // examine the extreme case where although part_i <0, L stays 0
        if (L == 0) {
            for (i = 0; i < N; i++)
                temp->child[i]->part = 0;
            return SimulateImageQuery_Nport(L, b, d);
        }
        t = SolveImageQuery_NPort_Aux(L, b, d);
        printf("AuxNP : %li, %lf %lf %i\n", L, t, reserv, valid);

        iter++;
        if (iter > ALG1_MAX_ITER) {
            PrintSolution();
            exit(1);
        }
    }

    valid = 1;
    if (iter > ALG1_MAX_ITER) return SimulateImageQuery_Nport(L, b, d); // if the nodes have been "truncated"

    // the closed-form solution is not accurate 
    return t;
}

//---------------------------------------------
// Used for comparing against the non-uniform distribution
// Load is transferred in case nodes are added or removed
// L is a reference because it can be modified

double Network::EqualDistrImageQuery_NPort(long &L, long b, double d) {
    Node *temp;
    int i, N = head->degree;
    double l, t, part, aux;
    double p[N], e[N];
    long L1;

    temp = head;
    l = temp->child[0]->l2par;

    L1 = L;
    for (i = 0; i < N; i++) {
        p[i] = temp->child[i]->power;
        e[i] = temp->child[i]->e0;
        L1 += (long) temp->child[i]->e0;
    }

    part = L1 / N;
    aux = 0;
    for (i = 0; i < N; i++)
        if (temp->child[i]->e0 > part) {
            aux += temp->child[i]->e0 - part; // this needs to be communicated to other nodes
            temp->child[i]->e0 = part;
            temp->child[i]->part = 0;
        }

    L += (long) ceil(aux); // this needs to be communicated

    for (i = 0; i < N; i++)
        if (temp->child[i]->e0 < part) {
            aux = part - temp->child[i]->e0;
            aux /= L;
            temp->child[i]->part = aux;
        }

    t = l * (temp->child[0]->part * L + b) + p[0] * (temp->child[0]->part * L + e[0]) + l * d;
    for (i = 1; i < N; i++) {
        aux = l * (temp->child[i]->part * L + b) + p[i] * (temp->child[i]->part * L + e[i]) + l * d;
        if (aux > t) t = aux;
    }
    return t;
}


//-----------------------------------------------------
// Generic pair of nodes best configuration finder
// returns 1, 2, 3 or 4 based on which arrangement is best
// By default blocktype is true
// L here is the load assigned to the two nodes collectively

int Network::FindBest(Node *pA, Node *pB, double L, long b, double d, double D, bool blocktype) {
    double p0, p1, e0, e1;
    double t3t4, t3t2, t3t1, t1t4, t1t2, t4t2, t1, t2, t3, t4, part0; // differences
    double denom, l0, l1;

    p0 = pA->power;
    e0 = pA->e0;
    p1 = pB->power;
    e1 = pB->e0;
    l0 = pA->l2par;
    l1 = pB->l2par;
    if (blocktype) {
        part0 = (p1 * (L + e1) - p0 * e0 + l1 * (L + b + d) + D) / (L * (p0 + p1 + l1));
        t1 = l0 * (part0 * L + b) + p0 * (part0 * L + e0) + l0*d;
        part0 = (p1 * (L + e1) - p0 * e0 - l0 * (b + d) - D) / (L * (p0 + p1 + l0));
        t2 = l1 * ((1 - part0) * L + b) + p1 * ((1 - part0) * L + e1) + l1*d;
        part0 = (p1 * (L + e1) - p0 * e0 + l1 * (L + b) - l0 * d - D) / (L * (p0 + p1 + l1));
        t3 = l0 * (part0 * L + b) + p0 * (part0 * L + e0) + l0 * d + l1 * d + D;
        part0 = (p1 * (L + e1) - p0 * e0 + l1 * d - l0 * b + D) / (L * (p0 + p1 + l0));
        t4 = l1 * ((1 - part0) * L + b) + p1 * ((1 - part0) * L + e1) + l1 * d + D + l0*d;
    } else // stream type tasks
    {
        part0 = (p1 * (L + e1) - p0 * e0 + l1 * (b + d) + D) / (L * (p0 + p1 - l0));
        t1 = l0 * (b + d) + p0 * (part0 * L + e0);
        part0 = (p1 * (L + e1) - p0 * e0 - L * l1 - l0 * (b + d) - D) / (L * (p0 + p1 - l0));
        t2 = l1 * (b + d) + p1 * ((1 - part0) * L + e1);
        part0 = (p1 * (L + e1) - p0 * e0 + l1 * b - l0 * d - D) / (L * (p0 + p1 - l0));
        t3 = l0 * b + p0 * (part0 * L + e0) + l0 * d + D + l1*d;
        part0 = (p1 * (L + e1) - p0 * e0 + l1 * (d - L) - l0 * b + D) / (L * (p0 + p1 - l0));
        t4 = l1 * b + p1 * ((1 - part0) * L + e1) + l1 * d + D + l0*d;
    }

    t3t4 = t3 - t4;
    t3t2 = t3 - t2;
    t3t1 = t3 - t1;
    t1t4 = t1 - t4;
    t1t2 = t1 - t2;
    t4t2 = t4 - t2;

    if (t3t4 <= 0) {
        if (t3t2 <= 0) {
            if (t3t1 <= 0) return 3;
            else return 1;
        } else {
            if (t1t2 <= 0) return 1;
            else return 2;
        }
    } else {
        if (t4t2 <= 0) {
            if (t1t4 <= 0) return 1;
            else return 4;
        } else {
            if (t1t2 <= 0) return 1;
            else return 2;
        }
    }
}
//-----------------------------------------------------

inline void Network::Swap(Node **pA, Node **pB) {
    Node *temp = *pA;
    *pA = *pB;
    *pB = temp;
}
//---------------------------------------------
// Calculates only the partitioning for the given distr. and collection sequences
// returns the running time
// N allows the calculation of the partitioning on a subset of nodes

double Network::SolveImageQueryPartition(long L, long b, double d, bool blocktype, int N) {
    if (N == 0) N = head->degree;

    // some arrays to simplify the code
    double p[N], l[N], e[N], A[N], B[N], D, part0, t;
    double prodA[N];
    int cs[N], ics[N];
    for (int i = 0; i < N; i++) {
        p[i] = head->child[i]->power;
        l[i] = head->child[i]->l2par;
        e[i] = head->child[i]->e0;
        ics[i] = head->child[i]->collection_order;
        cs[ head->child[i]->collection_order] = i;
    }

    // determine A and B constants
    if (blocktype)
        for (int i = 0; i < N - 1; i++) {
            if (ics[i] < ics[i + 1]) // configuration #3
            {
                D = 0;
                for (int j = ics[i] + 1; j < ics[i + 1]; j++)
                    D += l[cs[j]];
                D *= d;
                A[i] = p[i] / (p[i + 1] + l[i + 1]);
                B[i] = (p[i] * e[i] - p[i + 1] * e[i + 1] - l[i + 1] * b + l[i] * d + D) / (L * (p[i + 1] + l[i + 1]));
            } else // configuration #1
            {
                D = 0;
                for (int j = ics[i + 1] + 1; j < ics[i]; j++)
                    D += l[cs[j]];
                D *= d;
                A[i] = p[i] / (p[i + 1] + l[i + 1]);
                B[i] = (p[i] * e[i] - p[i + 1] * e[i + 1] - l[i + 1]*(b + d) - D) / (L * (p[i + 1] + l[i + 1]));
            }
        } else // find A and B for stream type
        for (int i = 0; i < N - 1; i++) {
            if (ics[i] < ics[i + 1]) // configuration #3
            {
                D = 0;
                for (int j = ics[i] + 1; j < ics[i + 1]; j++)
                    D += l[cs[j]];
                D *= d;
                A[i] = (p[i] - l[i]) / p[i + 1];
                B[i] = (p[i] * e[i] - p[i + 1] * e[i + 1] - l[i + 1] * b + l[i] * d + D) / (L * p[i + 1]);
            } else // configuration #1
            {
                D = 0;
                for (int j = ics[i + 1] + 1; j < ics[i]; j++)
                    D += l[cs[j]];
                D *= d;
                A[i] = (p[i] - l[i]) / p[i + 1];
                B[i] = (p[i] * e[i] - p[i + 1] * e[i + 1] - l[i + 1]*(b + d) - D) / (L * p[i + 1]);
            }
        }


    prodA[0] = A[0];
    for (int i = 1; i < N - 1; i++)
        prodA[i] = A[i] * prodA[i - 1];

    double nomin = 1, denom = 1;
    for (int i = 1; i < N; i++)
        denom += prodA[i - 1];

    for (int i = 0; i <= N - 2; i++) {
        double temp = 0;
        for (int j = i + 1; j <= N - 1; j++)
            temp += prodA[j - 1] / prodA[i];
        temp *= B[i];
        nomin -= temp;
    }
    part0 = nomin / denom;

    // find remaining parts and check validity
    double sum = part0;
    valid = 1;
    head->child[0]->part = part0;
    for (int i = 1; i <= N - 1; i++) {
        double temp = part0 * prodA[i - 1];

        for (int j = 0; j <= i - 1; j++)
            temp += B[j] * prodA[i - 1] / prodA[j];

        head->child[i]->part = temp;
        if (temp < 0) valid = 0;
        sum += temp;
    }
    if (fabs(sum - 1) > 0.0001) valid = 0;

    // find execution time
    t = 0;
    D = 0;
    for (int j = ics[0] + 1; j <= N - 1; j++)
        D += l[cs[j]];
    D *= d;

    if (blocktype)
        t = l[0]*(part0 * L + b) + p[0]*(part0 * L + e[0]) + D;
    else
        t = l[0] * b + p[0]*(part0 * L + e[0]) + D;

    if (t < 0) valid = 0; // <----  GB
    return t;
}
//---------------------------------------------
// generic single port
// Solves the generic problem by rearranging the nodes, without considering load shifts

double Network::SolveImageQuery_Aux(long L, long b, double d, bool blocktype, int *piter, int *pswap) {
    // static bool sort_effect=true;
    double t, D;
    int N = head->degree;
    Node **cpu;
    double orig_t, prev_t, best_t = 0;
    bool flag = true, best_valid;
    int iter = 0, swaps = 0;
    Node * distr_seq[N];
    int coll_seq[N];


    head->L = L;
    for (int i = 1; i < N; i++) // find the number of nodes excluding the ones that will receive no load
        if (head->child[i]->through == 3) {
            N = i;
            break;
        }

    ImageQuerySort(head, false, N); // initial sort according to cpu power

    cpu = head->child; // then fix the collection order

    for (int i = 0; i < head->degree; i++) // for all the nodes, even the "removed" ones
        cpu[i]->collection_order = N - i - 1;

    while (flag && (iter < 4 * N)) // through experimentation is was found that very little changes after 4*N passes. Changes if any are to 4th or 5th significant digit
    {
        flag = false;
        t = SolveImageQueryPartition(L, b, d, blocktype, N);
        if ((t < best_t && valid) || iter == 0 || best_t == 0) {
            best_t = t;
            best_valid = valid;
            for (int j = 0; j < N; j++) {
                distr_seq[j] = head->child[j];
                coll_seq[j] = head->child[j]->collection_order;
            }
        }

        if (iter == 0) {
            orig_t = t;
        }

        prev_t = t;

        //examine every pair of nodes
        for (int i = 0; i < N - 1; i++) {
            double localL = 0;
            localL = cpu[i]->part * L + cpu[i + 1]->part*L;

            int collect_slot1, collect_slot2;
            if (cpu[i]->collection_order > cpu[i + 1]->collection_order) {
                collect_slot1 = cpu[i + 1]->collection_order;
                collect_slot2 = cpu[i]->collection_order;
            } else {
                collect_slot1 = cpu[i]->collection_order;
                collect_slot2 = cpu[i + 1]->collection_order;
            }
            // find delay between collection phases
            // This is not an optimized code. Could be alot faster
            D = 0;
            for (int j = 0; j < N; j++)
                if (cpu[j]->collection_order > collect_slot1 &&
                        cpu[j]->collection_order < collect_slot2)
                    D += cpu[j]->l2par;
            D *= d;

            int best = FindBest(cpu[i], cpu[i + 1], localL, b, d, D, blocktype);

            switch (best) // fix the orders, distr. and coll.
            {
                case(1): // distr. seq.OK
                    if (cpu[i]->collection_order != collect_slot2) {
                        flag = true;
                        cpu[i]->collection_order = collect_slot2;
                        cpu[i + 1]->collection_order = collect_slot1;
                        swaps++;
                    }
                    break;
                case(2):
                    flag = true;
                    Swap(&(cpu[i]), &(cpu[i + 1]));
                    cpu[i]->collection_order = collect_slot1;
                    cpu[i + 1]->collection_order = collect_slot2;
                    swaps++;
                    break;
                case(3): // distr. seq.OK
                    if (cpu[i]->collection_order != collect_slot1) {
                        flag = true;
                        cpu[i]->collection_order = collect_slot1;
                        cpu[i + 1]->collection_order = collect_slot2;
                        swaps++;
                    }
                    break;
                default:
                    flag = true;
                    Swap(&(cpu[i]), &(cpu[i + 1]));
                    cpu[i]->collection_order = collect_slot2;
                    cpu[i + 1]->collection_order = collect_slot1;
                    swaps++;
                    break;
            }
        }
        iter++;
    }
    printf("Final: %lf %i    Best %lf %i\n", t, valid, best_t, best_valid);
    printf("N: %i Iter: %i Swaps: %i  Improv: %lf\n", N, iter, swaps, (orig_t - t) / orig_t);

    // enforce optimum order
    for (int j = 0; j < N; j++) {
        head->child[j] = distr_seq[j];
        head->child[j]->collection_order = coll_seq[j];
    }

    // restore the linked list
    //   head->next_n = head->child[0];
    head->next_n = distr_seq[0];
    for (int j = 0; j < N - 1; j++)
        head->child[j]->next_n = distr_seq[j + 1];
    if (N != head->degree) head->child[N - 1]->next_n = head->child[N];
    else head->child[N - 1]->next_n = NULL;

    SolveImageQueryPartition(L, b, d, blocktype, N); // solve again to get the optimum parts for the particular order

    if (piter != NULL) *piter = iter;
    if (pswap != NULL) *pswap = swaps;
    return best_t;
}
//---------------------------------------------
// Applies the image cache redistribution algorithm

double Network::SolveImageQuery_NPort_ST(long &L, long b, double d) // n-port stream-type type
{
    Node *temp;
    int i, k, M, N = head->degree;
    double p, l, t, aux;
    long added = 0;

    if (L == 0) added = L = 1;

    temp = head;
    head->L = L;

    t = SolveImageQuery_NPort_ST_Aux(L, b, d);

    if (valid == 1 && added == 0) return t;

    // Solution is invalid, examine L redistribution
    for (i = 0; i < N; i++)
        temp->child[i]->time = temp->child[i]->e0; // temporarily store previous e0

    //   double reserv=0;
    long reserv = 0;
    long laux;

    long iter = 0;
    while (valid == 0 || reserv != 0 || added != 0) // cause the added load to be fixed
    {
        reserv = 0;
        for (i = 0; i < N; i++)
            //         if(temp->child[i]->time != temp->child[i]->e0 || temp->child[i]->part<0)
            if (temp->child[i]->part < 0) {
                laux = (long) floor(temp->child[i]->e0 + temp->child[i]->part * L);
                if (laux < 0) {
                    valid = 0; // a subset of nodes should be used
                    return -1;
                } else {
                    //                 reserv +=  (temp->child[i]->e0 - laux);
                    //                 temp->child[i]->e0 = laux ;
                    int aux = (long) ceil(-temp->child[i]->part * L);
                    reserv += aux;
                    temp->child[i]->e0 -= aux;
                }
            }

        L += reserv;

        if (added > 0) {
            L -= added; // removes the small load that was added in order to get a first solution
            added = 0;
            reserv = 1; // just a token load to force recalculation
        }

        // examine the extreme case where although part_i <0, L stays 0
        if (L == 0) {
            for (i = 0; i < N; i++)
                temp->child[i]->part = 0;
            return SimulateImageQuery_Nport(L, b, d, true);
        }
        t = SolveImageQuery_NPort_ST_Aux(L, b, d);
        printf("AuxNP : %li, %lf %li %i\n", L, t, reserv, valid);

        iter++;
        if (iter > ALG1_MAX_ITER) {
            PrintSolution();
            exit(1);
        }
    }

    return t;
}
//---------------------------------------------
// Auxiliary function used by SolveImageQuery_NPort_ST to provide the partitioning calculations

double Network::SolveImageQuery_NPort_ST_Aux(long L, long b, double d) // n-port stream-type type
{
    int N = head->degree;
    long I = b; // it is assumed that I=b since b equals the size of an image
    double p[N], l[N], e[N], p0e0, part0, t;
    double nomin = 0, denom = 0;
    double total_cache = 0;

    for (int i = 0; i < N; i++) {
        p[i] = head->child[i]->power;
        l[i] = head->child[i]->l2par;
        e[i] = head->child[i]->e0;
        total_cache += e[i];
    }
    p0e0 = p[0] * e[0];

    for (int i = 0; i < N; i++)
        denom += 1.0 / p[i];
    denom *= p[0];

    if (total_cache == 0) // system initialization
    {
        for (int i = 1; i < N; i++)
            nomin += ((b + d + I)*(l[0] - l[i])) / p[i];
        nomin /= L;
        nomin = 1 - nomin;

        head->child[0]->part = part0 = nomin / denom;
        for (int i = 1; i < N; i++)
            head->child[i]->part = part0 * p[0] / p[i]+((l[0] - l[i])*(b + d + I)) / (L * p[i]);
        t = l[0]*(b + d + I) + p[0] * part0*L;
    } else // caches are not empty
    {
        for (int i = 1; i < N; i++)
            nomin += ((b + d)*(l[0] - l[i]) + (p0e0 - p[i] * e[i])) / p[i];
        nomin /= L;
        nomin = 1 - nomin;

        head->child[0]->part = part0 = nomin / denom;
        for (int i = 1; i < N; i++)
            head->child[i]->part = part0 * p[0] / p[i]+((l[0] - l[i])*(b + d)+(p0e0 - p[i] * e[i])) / (L * p[i]);
        t = l[0]*(b + d) + p[0]*(part0 * L + e[0]);
    }

    bool comm_bound = false;
    int comm_bound_count = 0;
    for (int i = 0; i < N; i++)
        if (p[i]*(head->child[i]->part * L + e[i]) < l[i] * L * head->child[i]->part) {
            comm_bound_count++;
        }

    // a switch to the comm. bound equations is done only if all the nodes are communicationally bound
    if (comm_bound_count == N) comm_bound = true;

    if (comm_bound) // assume comp. dominated execution
    {
        printf("COMM BOUND ");
        denom = 0;
        for (int i = 0; i < N; i++)
            denom += 1.0 / l[i];
        denom *= l[0];

        nomin = 0;
        for (int i = 1; i < N; i++)
            nomin += ((b + d)*(l[0] - l[i]) + I * (p[0] - p[i])) / l[i];
        nomin /= L;
        nomin = 1 - nomin;

        head->child[0]->part = part0 = nomin / denom;
        for (int i = 1; i < N; i++)
            head->child[i]->part = part0 * l[0] / l[i]+((l[0] - l[i])*(b + d) + I * (p[0] - p[i])) / (L * l[i]);

        t = 0;
        for (int i = 0; i < N; i++) {
            double temp = l[i]*(head->child[i]->part * L + b) + p[i] * I + l[i] * d;
            double temp2;
            if (e[i] == 0)
                temp2 = l[i]*(b + d + I) + p[i]*(head->child[i]->part * L);
            else
                temp2 = l[i]*(b + d) + p[i]*(head->child[i]->part * L + e[i]);
            if (temp < temp2) temp = temp2;
            if (t < temp) t = temp;
        }
    }

    // check validity
    valid = true;
    double total_s = 0;
    for (int i = 0; i < N; i++) {
        if (head->child[i]->part < 0) {
            valid = 0;
            break;
        }
        total_s += head->child[i]->part;
    }
    if (fabs(total_s - 1) > 0.0001) valid = 0;

    printf("ST_AUX %i %lf %i\n", valid, total_s, N);

    if (comm_bound_count > 0 && valid == 1) // just to be sure that the correct time is computed
        return SimulateImageQuery_Nport(L, b, d, true);
    return t;
}

//---------------------------------------------
// Generic 1-port solution. Uses a heuristic algorithm to find how to shift load, so that all parts are positive
// The equalizeLoad flag was introduced because for L==0 the stream-type tasks were causing continuous load
// shifts that undermined the efficiency to the effect of becoming slower than block-type tasks

double Network::SolveImageQuery(long &L, long b, double d, bool blocktype, bool firstcall, bool equalizeLoad) {
    Node *temp;
    int i, k, M, N = head->degree;
    double p, l, t, aux;
    long added = 0;

    if (L == 0 && equalizeLoad && !firstcall) // equalize the load so that subsequent load shifts do not happen
    {
        double totalCache = 0;
        long Ltemp;
        temp = head;
        for (i = 0; i < N; i++) {
            temp->child[i]->time = temp->child[i]->e0; // temporarily store previous e0
            totalCache += temp->child[i]->e0;
            temp->child[i]->e0 = 0;
        }
        Ltemp = (long) (totalCache + .495); // this must be an integer anyway
        EqualDistrImageQueryHeterog(Ltemp, b, d);

        double toBeComm = 0;
        for (i = 0; i < N; i++) {
            aux = temp->child[i]->time - temp->child[i]->part * Ltemp;
            if (aux >= 0) {
                temp->child[i]->e0 = temp->child[i]->part * Ltemp;
                temp->child[i]->part = 0;
            } else {
                temp->child[i]->e0 = temp->child[i]->time;
                toBeComm += -aux;
            }
        }

        // recalculate the parts for the nodes that will be receiving load
        if (toBeComm > 0) {
            for (i = 0; i < N; i++)
                if (temp->child[i]->part > 0)
                    temp->child[i]->part = (temp->child[i]->part * totalCache - temp->child[i]->e0) / toBeComm;
        }

        // there is some inaccuracy here because of the conversion
        L = (long) (toBeComm + .495); // modify reference L
        if (blocktype)
            return SimulateImageQuery(L, b, d);
        else
            return SimulateImageQuery_ST(L, b, d);
    }

    if (L == 0) added = L = 1;

    temp = head;
    for (i = 0; i < N; i++)
        temp->child[i]->through = 0;

    t = SolveImageQuery_Aux(L, b, d, blocktype);
    if (valid == 0 && firstcall) // in this case, a subset of nodes should be used
        return -1;

    if (valid == 0)
        for (i = 0; i < N; i++)
            temp->child[i]->time = temp->child[i]->e0; // temporarily store previous e0

    long reserv = 0;
    long laux;

    long iter = 0;
    int last_removed = N - 1;
    while (valid == 0 || reserv != 0 || added > 0) {
        reserv = 0;
        for (i = 0; i < N; i++)
            if (temp->child[i]->part < 0) {
                laux = (long) floor(temp->child[i]->e0 + temp->child[i]->part * L);
                if (laux < 0) {
                    valid = 0; // a subset of nodes should be used
                    return -1;
                } else {
                    int aux = (long) ceil(-temp->child[i]->part * L);
                    reserv += aux;
                    temp->child[i]->e0 -= aux;
                }
            }// cache is restored only if we do not have a stream type task
            else if ((temp->child[i]->time > temp->child[i]->e0) && (blocktype == true)) {
                double aux = temp->child[i]->part * L + temp->child[i]->e0;
                long diff;
                if (aux > temp->child[i]->time)
                    diff = (long) floor(temp->child[i]->time - temp->child[i]->e0);
                else
                    diff = (long) floor(aux - temp->child[i]->e0);
                reserv -= diff;
                temp->child[i]->e0 += diff;
                //          valid=0;
            }

        L += reserv;

        if (added > 0) {
            L -= added; // removes the small load that was added in order to get a first solution
            added = 0;
            reserv = 1;
        }

        // examine the extreme case where although part_i <0, L stays 0
        if (L == 0) {
            for (int i = 0; i < N; i++)
                temp->child[i]->part = 0;
            if (blocktype)
                return SimulateImageQuery(L, b, d);
            else
                return SimulateImageQuery_ST(L, b, d);
        }

        iter++;
        if (iter > ALG1_MAX_ITER) // this probably means that one of the nodes cannot be part of the distribution currently
        {
            if (last_removed < 1) // error checking
            {
                printf("Error in last_removed %li %i %li %li %i %i\n", iter, last_removed, L, reserv, valid, blocktype);
                for (int n = 0; n < N; n++)
                    printf("%s : %lf  %lf %lf\n", temp->child[n]->name, temp->child[n]->part, temp->child[n]->e0, temp->child[n]->time);
                valid = 0;
                return DBL_MAX;
            }
            temp->child[last_removed]->part = 0;
            temp->child[last_removed]->through = 3;
            last_removed--;
        }
        t = SolveImageQuery_Aux(L, b, d, blocktype); // not sorting the nodes gives inferior results
    }

    // reset through flag
    for (i = 0; i < N; i++)
        temp->child[i]->through = 0;
    valid = 1;

    if (N != head->degree) // if the nodes have been "truncated"
    { // the closed-form solution is not accurate
        if (blocktype)
            t = SimulateImageQuery(L, b, d);
        else
            t = SimulateImageQuery_ST(L, b, d);
    }
    return t;
}

//---------------------------------------------
// 1-port block-type tasks. Result collection phase accounted for in the LP formulation

double Network::SolveImageQuery_NInst(long L, long b, double d, int M) {
    assert(M <= MAX_INST);

    int N = head->degree;
    long I = b; // it is assumed that I=b since b equals the size of an image
    double p[N], l[N], e[N], D;
    int cs[N], ics[N];
    char pname[15];
    glp_prob *lp;
    int i, j, idx, ia[1 + N * N * M * M], ja[1 + N * N * M * M];
    double ar[1 + N * N * M * M], Ttotal, part[M][N];

    for (int j = 0; j < N; j++) {
        p[j] = head->child[j]->power;
        l[j] = head->child[j]->l2par;
        e[j] = head->child[j]->e0;
        ics[j] = head->child[j]->collection_order;
        cs[ head->child[j]->collection_order] = j;
    }

    lp = glp_create_prob();
    glp_set_prob_name(lp, "sample");
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, N * M); // name the slack variables. The last one that corresponds to
    // the normalization equation is LX_FIXED to 1
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            if (i == M - 1 && j == N - 1) break; // skip last slack var.
            sprintf(pname, "s_%i_%i", i + 1, j + 1);
            glp_set_row_name(lp, i * N + j + 1, pname);
            glp_set_row_bnds(lp, i * N + j + 1, GLP_LO, 0.0, 0.0);
        }
    }
    glp_set_row_name(lp, N*M, "s_norm"); // set last slack var
    glp_set_row_bnds(lp, N*M, GLP_FX, 1.0, 1.0);

    // specify the names and bounds of the variables
    glp_add_cols(lp, N * M);
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sprintf(pname, "part_%i_%i", i + 1, j + 1);
            glp_set_col_name(lp, i * N + j + 1, pname);
            glp_set_col_bnds(lp, i * N + j + 1, GLP_DB, 0.0, 1.0);
        }
    }

    // setup cost function
    // first clear everything
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            glp_set_obj_coef(lp, i * N + j + 1, 0);
        }

    for (j = 0; j < N; j++) {
        glp_set_obj_coef(lp, j + 1, l[j] * L);
    }
    for (i = 0; i < M; i++) {
        glp_set_obj_coef(lp, i * N + N - 1 + 1, p[N - 1] * L);
    }


    // coefficients. Initially all are cleared
    for (i = 0; i < N * M; i++) {
        for (j = 0; j < N * M; j++) {
            idx = i * N * M + j + 1;
            ia[idx] = i + 1;
            ja[idx] = j + 1;
            ar[idx] = 1e-300; // GLPK does not accept zeros
        }
    }
    //for normalization eq.
    for (i = 0; i < N * M; i++) {
        idx = (N * M - 1) * N * M + i + 1;
        ar[idx] = 1;
    }
    // remaining equations
    //---------------------------
    // for 1st installment
    for (j = 0; j < N; j++) {
        double lhs = 0;
        lhs = p[j] * e[j];
        idx = N * M * j + j + 1;
        ar[idx] = p[j] * L;
        int k = j + 1;
        while (k <= N - 1) {
            idx = N * M * j + k + 1;
            ar[idx] = -l[k] * L;
            lhs -= l[k] * b;
            k++;
        }

        k = 0;
        while (k <= j) {
            idx = N * M * j + N + k + 1;
            ar[idx] = -l[k] * L;
            k++;
        }
        glp_set_row_bnds(lp, j + 1, GLP_LO, -lhs, 0.0); // modification of the lower bound for the slack variable
    }
    //---------------------------
    // for installments 2 - M-2
    for (i = 1; i < M - 1; i++)
        for (j = 0; j < N; j++) {
            idx = N * M * (i * N + j) + i * N + j + 1; // N*M*(i*N+j) is the last element of the above rows
            ar[idx] = p[j] * L;
            int k = j + 1;
            while (k <= N - 1) {
                idx = N * M * (i * N + j) + i * N + k + 1;
                ar[idx] = -l[k] * L;
                k++;
            }
            k = 0;
            while (k <= j) {
                idx = N * M * (i * N + j) + (i + 1) * N + k + 1;
                ar[idx] = -l[k] * L;
                k++;
            }
        }

    //---------------------------
    // for last set of N-1 relations
    for (j = 0; j < N - 1; j++) {
        double lhs = 0;
        lhs = l[j + 1] * b;
        idx = N * M * ((M - 1) * N + j) + 0 * N + j + 1 + 1;
        ar[idx] = l[j + 1] * L;
        for (i = 0; i < M; i++) {
            idx = N * M * ((M - 1) * N + j) + i * N + j + 1 + 1;
            ar[idx] += p[j + 1] * L;
        }
        lhs += p[j + 1] * e[j + 1];

        for (i = 0; i < M; i++) {
            idx = N * M * ((M - 1) * N + j) + i * N + j + 1;
            ar[idx] = -p[j] * L;
        }
        lhs -= p[j] * e[j];

        int slot1 = MIN(ics[j], ics[j + 1]);
        int slot2 = MAX(ics[j], ics[j + 1]);

        D = 0;
        for (int k = slot1; k < slot2; k++)
            D += l[cs[k]];
        D *= d;
        if (ics[j] < ics[j + 1])
            lhs -= D;
        else
            lhs += D;

        glp_set_row_bnds(lp, (M - 1) * N + j + 1, GLP_LO, -lhs, 0.0); // modification of the lower bound for the slack variable
    }

    glp_load_matrix(lp, N * N * M*M, ia, ja, ar);
    glp_simplex(lp, NULL);
    Ttotal = glp_get_obj_val(lp);

    // add the constant terms to the cost function
    for (j = 0; j < N; j++)
        Ttotal += l[j] * b;
    Ttotal += p[N - 1] * e[N - 1] + l[N - 1] * d;

    valid = 1;
    double totLoad = 0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double temp = glp_get_col_prim(lp, i * N + j + 1);
            head->child[j]->mi_part[i] = temp;
            totLoad += temp;
            if (temp < 0) valid = 0;
        }
    }

    D = 0;
    for (int k = ics[N - 1] + 1; k < N; k++)
        D += l[cs[k]];
    D *= d;
    Ttotal += D;

    double aux = 0;
    for (j = 0; j < N; j++)
        aux += l[j] * L * head->child[j]->mi_part[0];
    for (i = 0; i < M; i++)
        aux += p[N - 1] * L * head->child[N - 1]->mi_part[i];

    // summing them up verification step
    double tmax = 0, commt = 0;
    for (j = 0; j < N; j++) {
        commt += l[j]*(b + head->child[j]->mi_part[0] * L);
        double ttemp = commt + p[j]*(head->child[j]->mi_part[0] * L + e[j]);
        for (i = 1; i < M; i++)
            ttemp += p[j] * head->child[j]->mi_part[i] * L;
        ttemp += l[j] * d;
        if (ttemp > tmax) tmax = ttemp;
    }

    if (fabs(totLoad - 1.0) > 0.000001) valid = 0;
    if (glp_get_status(lp) != GLP_OPT) // if p is substantially smaller than l, block type tasks cannot be occupied while load is being uploaded (no solution exists) 
    {
        valid = 0;
        printf("NO SOLUTION FOUND\nL: %li\n", L);
        printf("N: %i\n", N);
        printf("M: %i\n", M);
        PrintNet();
        glp_delete_prob(lp);
        return DBL_MAX; // indicate that the solution is not possible
    }
    glp_delete_prob(lp);
    return tmax;
}
//---------------------------------------------
// 1-port stream-type tasks
// Will modify the L and b parameters if e are zero.

double Network::SolveImageQuery_NInst_ST(long &L, long &b, double d, int M) {
    assert(M <= MAX_INST && M > 1);

    int N = head->degree;
    long I = b; // it is assumed that I=b since b equals the size of an image
    double p[N], l[N], e[N];
    int cs[N], ics[N];
    char pname[15];
    glp_prob *lp;
    int i, j, idx, ia[1 + N * N * M * M], ja[1 + N * N * M * M];
    double ar[1 + N * N * M * M], Ttotal, part[M][N], D;

    double noresidentflag = false;
    for (int j = 0; j < N; j++) {
        p[j] = head->child[j]->power;
        l[j] = head->child[j]->l2par;
        e[j] = head->child[j]->e0;
        if (e[j] == 0) noresidentflag = true;
        ics[j] = head->child[j]->collection_order;
        cs[ head->child[j]->collection_order] = j;
    }
    if (noresidentflag) {
        L -= N*I;
        //assert(L>0);
        if (L <= 0) {
            valid = false;
            return 0;
        }
        b += I;
        for (int j = 0; j < N; j++) {
            e[j] += I;
            head->child[j]->e0 += I;
        }
    }

    lp = glp_create_prob();
    glp_set_prob_name(lp, "sample");
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, N * M); // name the slack variables. The last one that corresponds to 
    // the normalization equation is LX_FIXED to 1
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            if (i == M - 1 && j == N - 1) break; // skip last slack var.
            sprintf(pname, "s_%i_%i", i + 1, j + 1);
            glp_set_row_name(lp, i * N + j + 1, pname);
            glp_set_row_bnds(lp, i * N + j + 1, GLP_LO, 0.0, 0.0);
        }
    }
    glp_set_row_name(lp, N*M, "s_norm"); // set last slack var
    glp_set_row_bnds(lp, N*M, GLP_FX, 1.0, 1.0);

    // specify the names and bounds of the variables
    glp_add_cols(lp, N * M);
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sprintf(pname, "part_%i_%i", i + 1, j + 1);
            glp_set_col_name(lp, i * N + j + 1, pname);
            glp_set_col_bnds(lp, i * N + j + 1, GLP_DB, 0.0, 1.0);
        }
    }

    // setup cost function
    // first clear everything
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            glp_set_obj_coef(lp, i * N + j + 1, 0);
        }
    // get the coefficients of the proper unknowns
    for (j = 0; j < N - 1; j++) {
        glp_set_obj_coef(lp, j + 1, l[j] * L);
    }
    for (i = 0; i < M; i++) {
        glp_set_obj_coef(lp, i * N + N - 1 + 1, p[N - 1] * L);
    }


    // coefficients. Initially all are cleared
    for (i = 0; i < N * M; i++) {
        for (j = 0; j < N * M; j++) {
            idx = i * N * M + j + 1;
            ia[idx] = i + 1;
            ja[idx] = j + 1;
            ar[idx] = 1e-300; // GLPK does not accept zeros
        }
    }
    //for normalization eq.
    for (i = 0; i < N * M; i++) {
        idx = (N * M - 1) * N * M + i + 1;
        ar[idx] = 1;
    }
    // remaining equations
    //---------------------------
    // for 1st installment
    for (j = 0; j < N; j++) {
        double lhs = 0;
        lhs = p[j] * e[j];
        idx = N * M * j + j + 1;
        ar[idx] = (p[j] - l[j]) * L;
        int k = j + 1;
        while (k <= N - 1) {
            idx = N * M * j + k + 1;
            ar[idx] = -l[k] * L;
            lhs -= l[k] * b;
            k++;
        }

        k = 0;
        while (k < j) {
            idx = N * M * j + N + k + 1;
            ar[idx] = -l[k] * L;
            k++;
        }
        lhs -= l[j] * I;

        glp_set_row_bnds(lp, j + 1, GLP_LO, -lhs, 0.0); // modification of the lower bound for the slack variable
    }
    //---------------------------
    // for installments 2 - M-2
    for (i = 1; i < M - 1; i++)
        for (j = 0; j < N; j++) {
            idx = N * M * (i * N + j) + i * N + j + 1; // N*M*(i*N+j) is the last element of the above rows
            ar[idx] = (p[j] - l[j]) * L;
            int k = j + 1;
            while (k <= N - 1) {
                idx++;
                ar[idx] = -l[k] * L;
                k++;
            }
            k = 0;
            while (k < j) {
                idx++;
                ar[idx] = -l[k] * L;
                k++;
            }
        }

    //---------------------------
    // for last set of N-1 relations
    for (j = 0; j < N - 1; j++) {
        double lhs = 0;
        lhs = l[j + 1] * b;
        idx = N * M * ((M - 1) * N + j) + 0 * N + j + 1;
        ar[idx] = l[j] * L;
        for (i = 0; i < M; i++) {
            idx = N * M * ((M - 1) * N + j) + i * N + j + 1 + 1;
            ar[idx] = p[j + 1] * L;
        }
        lhs += p[j + 1] * e[j + 1];

        for (i = 0; i < M; i++) {
            idx = N * M * ((M - 1) * N + j) + i * N + j + 1;
            ar[idx] -= p[j] * L;
        }
        lhs -= p[j] * e[j];

        int slot1 = MIN(ics[j], ics[j + 1]);
        int slot2 = MAX(ics[j], ics[j + 1]);

        D = 0;
        for (int k = slot1; k < slot2; k++)
            D += l[cs[k]];
        D *= d;
        if (ics[j] < ics[j + 1])
            lhs -= D;
        else
            lhs += D;

        glp_set_row_bnds(lp, (M - 1) * N + j + 1, GLP_LO, -lhs, 0.0); // modification of the lower bound for the slack variable
    }

    glp_load_matrix(lp, N * N * M*M, ia, ja, ar);
    glp_simplex(lp, NULL);
    Ttotal = glp_get_obj_val(lp);

    // add the constant terms to the cost function
    for (j = 0; j < N; j++)
        Ttotal += l[j] * b;
    Ttotal += p[N - 1] * e[N - 1] + l[N - 1] * d;

    valid = 1;
    double totLoad = 0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double temp = glp_get_col_prim(lp, i * N + j + 1);
            head->child[j]->mi_part[i] = temp;
            totLoad += temp;
            if (temp < 0) valid = 0;
        }
    }

    D = 0;
    for (int k = ics[N - 1] + 1; k < N; k++)
        D += l[cs[k]];
    D *= d;
    Ttotal += D;

    if (fabs(totLoad - 1.0) > 0.000001) valid = 0;
    if (glp_get_status(lp) != GLP_OPT) // if p is substantially smaller than l, block type tasks cannot be occupied while load is being uploaded (no solution exists)
    {
        valid = 0;
        printf("L: %li\n", L);
        printf("N: %i\n", N);
        printf("M: %i\n", M);
        PrintNet();
        glp_delete_prob(lp);
        return DBL_MAX;
    }
    glp_delete_prob(lp);
    return Ttotal;
}
//---------------------------------------------
// N-port block-type tasks

double Network::SolveImageQuery_NInst_NPort(long L, long b, double d, int M) {
    assert(M <= MAX_INST);

    int N = head->degree;
    double p[N], l[N], e[N];
    char pname[15];
    int i, j;
    double Ttotal, part00;
    double gamma[N], delta[N], epsilon[N];

    for (int i = 0; i < N; i++) {
        p[i] = head->child[i]->power;
        l[i] = head->child[i]->l2par;
        e[i] = head->child[i]->e0;
    }

    for (int j = 0; j < N; j++) {
        double aux = p[j] / l[j];
        if (aux == 1)
            gamma[j] = M;
        else
            gamma[j] = (pow(aux, M) - 1) / (aux - 1);
    }

    delta[0] = 1;
    for (int j = 1; j < N; j++)
        delta[j] = (l[0] + p[0] * gamma[0]) / (l[j] + p[j] * gamma[j]);

    epsilon[0] = 0;
    for (int j = 1; j < N; j++)
        epsilon[j] = ((b + d)*(l[0] - l[j]) + p[0] * e[0] * gamma[0] - p[j] * e[j] * gamma[j]) / (L * (l[j] + p[j] * gamma[j]));

    double sum1 = 0, sum2 = 0, sum3 = 0;
    for (j = 0; j < N; j++) {
        sum1 += epsilon[j] * gamma[j];
        sum3 += delta[j] * gamma[j];
    }

    for (j = 1; j < N; j++)
        sum2 += e[j]*(gamma[j] - 1);
    sum2 /= L;

    // calculate first installment first
    valid = 1;
    part00 = (1 - sum1 - sum2) / sum3;
    head->child[0]->mi_part[0] = part00;
    double totLoad = part00;
    for (j = 1; j < N; j++) {
        double temp = part00 * delta[j] + epsilon[j];
        head->child[j]->mi_part[0] = temp;
        totLoad += temp;
        if (temp < 0) valid = 0;
    }

    // find the remaining parts
    for (i = 1; i < M; i++) {
        for (j = 0; j < N; j++) {
            double temp = head->child[j]->mi_part[i - 1] * p[j] / l[j];
            head->child[j]->mi_part[i] = temp;
            totLoad += temp;
            if (temp < 0) valid = 0;
        }
    }
    if (fabs(totLoad - 1.0) > 0.000001) valid = 0;

    Ttotal = l[0]*(part00 * L + b);
    for (i = 1; i < M; i++)
        Ttotal += l[0] * head->child[0]->mi_part[i] * L;
    Ttotal += p[0] * head->child[0]->mi_part[M - 1] * L + l[0] * d;

    return Ttotal;
}

//-----------------------------------------------------------------------------
// Same as FindBest but for M installments

int Network::FindBest2(Node *pA, Node *pB, long L, long b, double d, double D, int M, bool blocktype) {
    Network n1, n2;
    double t1, t2, t3, t4, t3t4, t3t2, t3t1, t1t4, t1t2, t4t2;
    if (L == 0) return 0;
    //L=1; // avoid crashes for lightly loaded nodes

    long Ltemp = L, btemp = b;

    Node *x = n1.InsertNode((char*) "LON", 1, 0, (char*) "", 0);
    n1.InsertNode((char*) "P1", pA->power, pA->e0, x, pA->l2par, 1);
    n1.InsertNode((char*) "P2", pB->power, pB->e0, x, pB->l2par, 1);

    x = n2.InsertNode((char*) "LON", 1, 0, (char*) "", 0);
    n2.InsertNode((char*) "P2", pB->power, pB->e0, x, pB->l2par, 1);
    n2.InsertNode((char*) "P1", pA->power, pA->e0, x, pA->l2par, 1);

    n1.head->child[0]->collection_order = 1;
    n1.head->child[1]->collection_order = 0;

    n2.head->child[0]->collection_order = 0;
    n2.head->child[1]->collection_order = 1;

    if (blocktype) {
        t1 = n1.SolveImageQuery_NInst(Ltemp, btemp, d, M);
        t2 = n2.SolveImageQuery_NInst(Ltemp, btemp, d, M);
    } else {
        t1 = n1.SolveImageQuery_NInst_ST(Ltemp, btemp, d, M);
        Ltemp = L;
        btemp = b;
        t2 = n2.SolveImageQuery_NInst_ST(Ltemp, btemp, d, M);
    }

    n1.head->child[0]->collection_order = 0;
    n1.head->child[1]->collection_order = 1;

    n2.head->child[0]->collection_order = 1;
    n2.head->child[1]->collection_order = 0;

    if (blocktype) {
        t3 = n1.SolveImageQuery_NInst(Ltemp, btemp, d, M);
        t4 = n2.SolveImageQuery_NInst(Ltemp, btemp, d, M);
    } else {
        Ltemp = L;
        btemp = b;
        t3 = n1.SolveImageQuery_NInst_ST(Ltemp, btemp, d, M);
        Ltemp = L;
        btemp = b;
        t4 = n2.SolveImageQuery_NInst_ST(Ltemp, btemp, d, M);
    }

    t3t4 = t3 - t4;
    t3t2 = t3 - t2;
    t3t1 = t3 - t1;
    t1t4 = t1 - t4;
    t1t2 = t1 - t2;
    t4t2 = t4 - t2;

    if (t3t4 <= 0) {
        if (t3t2 <= 0) {
            if (t3t1 <= 0) return 3;
            else return 1;
        } else {
            if (t1t2 <= 0) return 1;
            else return 2;
        }
    } else {
        if (t4t2 <= 0) {
            if (t1t4 <= 0) return 1;
            else return 4;
        } else {
            if (t1t2 <= 0) return 1;
            else return 2;
        }
    }
}
//---------------------------------------------
// generic single port
// Solves the generic problem by rearranging the nodes, without considering load shifts
// The last two parameters are for collecting statistics only

double Network::SolveImageQuery_NInst_Aux(long &L, long &b, double d, int M, bool blocktype, int *piter, int *pswap) {
    double t, D;
    int N = head->degree;
    Node **cpu;
    double orig_t, prev_t, best_t = DBL_MAX;
    bool flag = true, best_valid;
    int iter = 0, swaps = 0;
    Node * distr_seq[N];
    int coll_seq[N];
    long Ltemp, btemp;

    head->L = L;
    ImageQuerySort(head, false, N); // initial sort according to cpu power

    cpu = head->child; // then fix the collection order

    for (int i = 0; i < head->degree; i++) // for all the nodes, even the "removed" ones
        cpu[i]->collection_order = i;

    while (flag && (iter < 4 * N)) // through experimentation is was found that very little changes after 4*N passes. Changes if any are to 4th or 5th significant digit
    {
        flag = false;

        if (blocktype)
            t = SolveImageQuery_NInst(L, b, d, M);
        else
            t = SolveImageQuery_NInst_ST(L, b, d, M);

        if (t < best_t && valid) {
            best_t = t;
            best_valid = valid;
            for (int j = 0; j < N; j++) {
                distr_seq[j] = head->child[j];
                coll_seq[j] = head->child[j]->collection_order;
            }
        }

        if (iter == 0)
            orig_t = t;

        prev_t = t;

        //examine every pair of nodes
        for (int j = 0; j < N - 1; j++) {
            double localL = 0;
            for (int i = 0; i < M; i++)
                localL = cpu[j]->mi_part[i] + cpu[j + 1]->mi_part[i];
            localL *= Ltemp;

            int collect_slot1, collect_slot2;
            if (cpu[j]->collection_order > cpu[j + 1]->collection_order) {
                collect_slot1 = cpu[j + 1]->collection_order;
                collect_slot2 = cpu[j]->collection_order;
            } else {
                collect_slot1 = cpu[j]->collection_order;
                collect_slot2 = cpu[j + 1]->collection_order;
            }
            // find delay between collection phases
            // This is not an optimized code. Could be alot faster
            D = 0;
            for (int k = 0; k < N; k++)
                if (cpu[k]->collection_order > collect_slot1 &&
                        cpu[k]->collection_order < collect_slot2)
                    D += cpu[k]->l2par;
            D *= d;

            int best = FindBest2(cpu[j], cpu[j + 1], (long) localL, btemp, d, D, M, blocktype);

            switch (best) // fix the orders, distr. and coll.
            {
                case(0): break; // do not do anything. Probably localL is zero
                case(1): // distr. seq.OK
                    if (cpu[j]->collection_order != collect_slot2) {
                        flag = true;
                        cpu[j]->collection_order = collect_slot2;
                        cpu[j + 1]->collection_order = collect_slot1;
                        swaps++;
                    }
                    break;
                case(2):
                    flag = true;
                    Swap(&(cpu[j]), &(cpu[j + 1]));
                    cpu[j]->collection_order = collect_slot1;
                    cpu[j + 1]->collection_order = collect_slot2;
                    swaps++;
                    break;
                case(3): // distr. seq.OK
                    if (cpu[j]->collection_order != collect_slot1) {
                        flag = true;
                        cpu[j]->collection_order = collect_slot1;
                        cpu[j + 1]->collection_order = collect_slot2;
                        swaps++;
                    }
                    break;
                default:
                    flag = true;
                    Swap(&(cpu[j]), &(cpu[j + 1]));
                    cpu[j]->collection_order = collect_slot2;
                    cpu[j + 1]->collection_order = collect_slot1;
                    swaps++;
                    break;
            }
        }
        iter++;
    }

    if (best_t == DBL_MAX) // no valid solution for given setup
    {
        valid = 0;
        return best_t;
    }

    // enforce optimum order
    for (int j = 0; j < N; j++) {
        head->child[j] = distr_seq[j];
        head->child[j]->collection_order = coll_seq[j];
    }

    // restore the linked list
    //   head->next_n = head->child[0];
    head->next_n = distr_seq[0];
    for (int j = 0; j < N - 1; j++)
        head->child[j]->next_n = distr_seq[j + 1];
    if (N != head->degree) head->child[N - 1]->next_n = head->child[N];
    else head->child[N - 1]->next_n = NULL;

    // solve again to get the optimum parts for the particular order  
    if (blocktype)
        t = SolveImageQuery_NInst(L, b, d, M);
    else
        t = SolveImageQuery_NInst_ST(L, b, d, M);

    if (piter != NULL) *piter = iter;
    if (pswap != NULL) *pswap = swaps;
    return best_t;
}
//-----------------------------------------------------------------------------------------
// Does an exhaustive search to find the best distribution/collection ordering. VERY TIME CONSUMING
// Complexity is (N!)^2
// Works for both single and multi-installment strategies

double Network::SolveImageQuery_NInst_Optimum(long &L, long &b, double d, int M, bool blocktype, double *worstp) {
    int N = head->degree;
    vector < Node *> v;
    vector <int> collv;
    vector < Node *> best_v;
    vector <int> best_collv;
    double t, bestT = DBL_MAX, worstT = DBL_MIN;

    for (int i = 0; i < N; i++)
        v.push_back(head->child[i]); // prepare the vector for the permutations

    sort(v.begin(), v.end());
    do {
        for (int j = 0; j < N; j++)
            head->child[j] = v[j];

        for (int j = 0; j < N; j++)
            collv.push_back(j);

        do {
            for (int j = 0; j < N; j++)
                head->child[j]->collection_order = collv[j];

            long Ltemp = L, btemp = b;
            if (M == 1)
                t = SolveImageQueryPartition(L, b, d, blocktype, N);
            else {
                if (blocktype)
                    t = SolveImageQuery_NInst(Ltemp, btemp, d, M);
                else
                    t = SolveImageQuery_NInst_ST(Ltemp, btemp, d, M);
            }

            //        if(t < bestT && >0)
            if (t < bestT && valid) {
                bestT = t;
                best_v = v;
                best_collv = collv;
            }

            if (t > worstT)
                worstT = t;
        }        while (next_permutation(collv.begin(), collv.end()));
        collv.clear();
    } while (next_permutation(v.begin(), v.end()));

    // in case no solution exists
    if (bestT == DBL_MAX) {
        valid = 0;
        return bestT;
    }

    for (int j = 0; j < N; j++)
        head->child[j] = best_v[j];
    for (int j = 0; j < N; j++)
        head->child[j]->collection_order = best_collv[j];

    //solve again to get the proper part_ij
    if (M == 1)
        SolveImageQueryPartition(L, b, d, blocktype, N);
    else {
        if (blocktype)
            SolveImageQuery_NInst(L, b, d, M);
        else
            SolveImageQuery_NInst_ST(L, b, d, M);
    }
    if (worstp != NULL) *worstp = worstT;
    return bestT;
}


