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


#ifndef _SCHEDULE_CORE_H_
#define _SCHEDULE_CORE_H_


#ifndef MAX_NODE_DEGREE
#define MAX_NODE_DEGREE 4 /* the maximum number of children of a Node */
#endif

extern int sched_lib_max_node_degree; // this controls the maximum degree of a node
// initialized to MAX_NODE_DEGREE

#ifndef MAX_NUM_NODES
#define MAX_NUM_NODES 200 /* used for handling networks by Greedy and AuxOptimum */
#endif

#ifndef MAX_INST
#define MAX_INST 100 /* used for multi-installment strategies in image query problems */
#endif

#define MAX_NAME 8
#define MAX_FLOAT FLT_MAX

#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#define MIN(a,b)  (((a) < (b)) ? (a) : (b))

#define ALG1_MAX_ITER 20

// times are start_to_receive, finished_rec, st_2_process,fin_proc
// ready_2_send,st_2_send, fin_sending
struct timing {
    double STR, FR, STP, FPRO, RTS, STS, FS;
};

struct Node {
    char name[MAX_NAME + 1];
    unsigned int ID; /* for same purpose as name */
    double power;
    double part; /* the part of the computation assigned to Node */
    double L; /* the load assigned to tree rooted by this node */
    long Lint; /* quantized load assigned to this node (not to subtree) */
    double e0; /* the constant part of the computation */
    double l2par; /* link to parent. Redundant but helpful */

    double aggregate_p; /* computed for inside Nodes */
    double aggregate_e;

    int degree; /* the number of children of this Node */
    Node *parent; /* the parent Node. if NULL this is the host Node */
    double link[MAX_NODE_DEGREE]; /* arrays for the subtree rooted at each Node */
    Node *child[MAX_NODE_DEGREE];
    char visited; // flag
    bool fe; // flag. 1 means that node is equipped with a front end
    Node *next_n;

    double time; // used in PrintTimes* and Quantify
    timing t; // for Simulation* methods

    char through; // when 1 then this processor sends all its load to its children
    // when 2 then the through flag cannot be reset

    int collection_order; // used in the single-port image query
    double mi_part[MAX_INST];

    bool operator<(const Node & x) const // convenience operator for generating node permutations
    {
        return power > x.power;
    }
};

//**************************************************************

class Network // a class for a bunch of Nodes forming a tree
{
private:
    Node netnode[MAX_NUM_NODES]; // pre-allocation simplifies memory usage and avoids memory handling overhead and leakage
    char node_usage[MAX_NUM_NODES + 1];

    double max_dL; // set by Quantify
    double min_dL;
    double sum_dL;
    char best_quant;
    double quant_per_better;
    unsigned long num_elem_dL;
    double max_gain; //max possible gain if t2 was used instead of t4
    char improved_flag; // used for gathering statistics. Set if Greedy2 improves Greedy1


    // generic methods
    void AuxInsertNode(char *c, double speed, double e, double link2parent);
    inline void Swap(Node **pA, Node **pB);
    bool CheckSolution();
    friend Node *DuplicateNode(Node *x);
    void InsertDuplicateNode(Node *x);
    void RemoveDuplicateNode(void);

    void AuxQuantify1();
    void AuxQuantify2();

    // for query processing
    void QueryAggregate(Node *temp, double b, double d);
    int QueryPart(Node *temp, double b, double d);
    void GreedyRev(bool ImageOrQuery, Network &test, long L, double ab, double cd);
    void UniformGreedy(bool ImageOrQuery, Network &test, long L, double ab, double cd);

	// for image processing
    friend bool CheckT4(Node *temp,double a);
    void ImageAggregate(Node *temp, double a, double c);
    int ImagePart(Node *temp, double a, double c);

    // the following methods search for a better utilization of
    // a given network for  a load or range of loads
    void AuxGreedy(bool ImageOrQuery, long start_L, long end_L, double a, double c);
    void Aux2Greedy(bool ImageOrQuery, Network &test, long L, double ab, double cd);

    void AuxOptimum(bool, Network &, long, double, double);

	// for image query processing
    double SolveImageQueryHomo_Aux(long L, long b, double d, bool sortflag = true);
    double SolveImageQuery_NPort_Aux(long L, long b, double d); // L is not redistributed
    int FindBest(Node *pA, Node *pB, double L, long b, double d, double D, bool blocktype = true);
    // applies the node rearrangement. Generic for 1-port, both block and stream type
    double SolveImageQuery_Aux(long L, long b, double d, bool blocktype = true, int *piter = NULL, int *pswap = NULL); // generic single port
    double SolveImageQuery_NPort_ST_Aux(long L, long b, double d); // n-port stream-type type, auxiliary function
    double SolveImageQueryPartition(long L, long b, double d, bool blocktype = true, int N = 0); // partition calculation for a specific sequence. This is a helper function, called by SolveImageQuery
    // for multi-installment
    int FindBest2(Node *pA, Node *pB, long L, long b, double d, double D, int M, bool blocktype = true);
    double SolveImageQuery_NInst_Aux(long &L, long &b, double d, int M, bool blocktype = true, int *piter = NULL, int *pswap = NULL); // Generic 1-port method. This is the one to be called


public:
    Node *head;
    Node *tail;
    Node *redundant; // used to keep the nodes that the Quantify procedures might
                     // clip off. These can be used later by ReUse... function

    char clipping; // Set before calling Quantify to 1 if clipping
                   // When Quantify returns is signifies whether the nodes need to be reduced

    char valid; //used to check if for a given L,an optimum distribution is possible

    bool t4_holds; // set by SolveImage

    Network();
    ~Network() {};
    Network& operator =(Network& x);
    Node* InsertNode(char *c, double speed, double e, Node *parent, double link2parent, bool fe = false, bool thru = false);
    Node* InsertNode(char *c, double speed, double e, char *parent, double link2parent, bool fe = false, bool thru = false);
    void PrintSolution(bool quantized = 0);
    void GenerateRandomTree(bool ImageOrQuery, int N,float min_p, float max_p, float min_l, float max_l, float min_e, float max_e, bool full_tree, bool all_fe);
    void ReUseRandomTree(bool ImageOrQuery, float min_p, float max_p, float min_l, float max_l, float min_e, float max_e, bool full_tree, bool all_fe);
    friend void QuerySort(Node *x);
    friend void ImageSort(Node *x);

    // the following functions accommodate networks of mixed processors
    void SolveQuery(long L, double b, double d, bool plain = 0);
    void SolveImage(long L, double a, double c, bool plain = 0);

    // the following methods search for a better utilization of 
    // a given network for  a load or range of loads
    void GreedyQuery(long start_L, long end_L, double b, double d);
    void GreedyImage(long start_L, long end_L, double a, double c);
    void GreedyQuery(Network &, long L, double b, double d);
    void GreedyImage(Network &, long L, double a, double c);
    void GreedyQueryRev(Network &test, long L, double b, double d);
    void GreedyImageRev(Network &test, long L, double a, double c);
    void UniformImageGreedy(Network &test, long L, double a, double c);
    void UniformQueryGreedy(Network &test, long L, double b, double d);

    void OptimumQuery(Network &, long L, double b, double d);
    void OptimumImage(Network &, long L, double a, double c);

    void TransQueryDistr(Network &, Network &);

    // rounds off the fractional parts of the partial loads
    double Quantify(bool ImageOrQuery, double ab, double cd); // returns the estimated run time
    void ClipIdleNodes();

    // the following print the detailed execution times for each node
    void PrintTimes4Image(double a, double c);
    void PrintTimes4Query(double b, double d);

    // for not arbitrary divisible tasks
    double SimulateQuery(double b, double d, bool output = 1);
    double SimulateImage(double a, double c, bool output = 1);

    void PrintNet(void);

    // for "image query" partitioning
    // 1-port communications and block-type tasks are the default. For every other option there is a specific addition to the method name
    // L is supposed to be the number of images and thus all the other parameters 
    // e.g. p, l, e, b, and d are expressed in terms of image units
    double SolveImageQueryHomogeneous(long &L, long b, double d, bool firstcall); // L is a reference because it can be modified
    double SimulateImageQuery(long L, long b, double d);
    double SimulateImageQuery_ST(long L, long b, double d);
    void ImageQueryHomoEmbed(long L);
    void ImageQueryEmbed(long L);
    double EqualDistrImageQuery(long &L, long b, double d); // used for comparison purposes
    double EqualDistrImageQueryHeterog(long &L, long b, double d);
    // for N-port communications
    double SolveImageQuery_NPort(long &L, long b, double d); // includes Algorithm 1
    double SimulateImageQuery_Nport(long L, long b, double d, bool isstream = false);
    double EqualDistrImageQuery_NPort(long &L, long b, double d);

    double SolveImageQuery(long &L, long b, double d, bool blocktype = true, bool firstcall = false, bool equalizeLoad = true);

    double SolveImageQuery_NPort_ST(long &L, long b, double d); // n-port stream-type type


    // multi-installment image quary solutions
    double SolveImageQuery_NInst(long L, long b, double d, int M); // 1-port block-type tasks
    double SolveImageQuery_NInst_ST(long &L, long &b, double d, int M); // 1-port stream-type tasks
    double SolveImageQuery_NInst_NPort(long L, long b, double d, int M); // N-port block-type tasks
    double SolveImageQuery_NInst_Optimum(long &L, long &b, double d, int M, bool blocktype = true, double *worstp = NULL); // exhaustive search for best distr./collection ordering. Works for both single and multi-installment strategies
    void ImageQueryEmbed_Ninst(long L, int M);
};

#endif
