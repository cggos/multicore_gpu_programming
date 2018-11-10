typedef struct PartStr 
{
  int x;
  int y;
} PartStr;


int check_proxim(int *pic, int cols, int x, int y);
int three_way();
void dla_init(int *pic, int rows, int cols, int particles, int init_seed);

/* The arrays pic and pic2 are assumed of being 2 rows and 2 columns
 * wider than necessary, to simplify boundary condition checking
 * 
 * pic2 is used for storing the updated simulation state
 * The caller should altetnate between pic and pic2 between 
 * invocations of the function.
 */
int *dla_evolve(int *pic, int *pic2, int rows, int cols);

void dla_init_plist (int *pic, int rows, int cols, PartStr *p, int particles, int init_seed);
void dla_evolve_plist (int *pic, int rows, int cols, PartStr *p, int *particles, PartStr *changes, int *numChanges);
void apply_changes(int *pic, int rows, int cols, PartStr *changes, int numChanges);