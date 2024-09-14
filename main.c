#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void printf_color_reset() {printf("\033[1;0m"); return;} // Reset (0)
void printf_color_red() {printf("\033[1;31m"); return;} // Red (31)

unsigned char rand_init = 0;
unsigned char rand_binary() {return (unsigned char)(rand() % 2);}
double rand_double_01() {double prec = 1000000000000000.0; if(rand_init == 0) {srand(time(0)); rand_init = 1;} int residual_int = (int)(prec); double residual = prec - ((double)(residual_int)); return ( ((double)( rand() % (residual_int + 1) )) + (residual * ((double)rand_binary())) ) / prec;} // from 0 to 1
double rand_double_11() {return (1.0 - (2.0 * ((double)rand_binary()))) * rand_double_01();} // from -1 to 1
double rand_double(double from, double to) {
    if(from == -1.0 && to == 1.0) return rand_double_11(); else if(from == 0.0 && to == 1.0) return rand_double_01();
    double prec = 10000000000.0; from *= prec; to *= prec; if(rand_init == 0) {srand(time(0)); rand_init = 1;} 
    int residual_int = (int)(to - from); double residual = to - from - ((double)(residual_int)); 
    return ( ((double)( rand() % (residual_int + 1) )) + from + (residual * ((double)rand_binary())) ) / prec;}
double exp(double x) {unsigned char precision = 30; double e = 1.0; double y = 1.0; double fact = 1.0; unsigned char i; for(i=1; i<=precision; i++) {y *= x; fact *= (double)i; e += (y/fact);} return e;}
double tanh(double x) {double e = exp(2.0 * x); return (e - 1.0) / (e + 1.0);}
double sigmoid(double x) {return 1.0 / (1.0 + exp(-1.0 * x));}

// ----------------------------------------------------------------------------------------------------------------

// garbage collector
typedef struct memory_node {void *data; struct memory_node *next;} memory_node; typedef memory_node *memory;
memory mem; memory mem_last; double mem_size; // mem_size in MiB
void memory_init() {mem = NULL; mem_last = NULL; mem_size = 0.0;}
void* memory_add(size_t size) {
    void *data = malloc(size); if(data != NULL) {
        size_t m = sizeof(memory_node);
        if(mem == NULL) {mem = (memory)malloc(m); if(mem != NULL) {mem->data = data; mem_last = mem; mem_size = ((double)(size + m)) / 1048576.0;}}
        else {mem_last->next = (memory)malloc(m); if(mem_last->next != NULL) {mem_last = mem_last->next; mem_last->data = data; mem_size += ((double)(size + m)) / 1048576.0;}}
    } return data;}
void memory_free_all() {memory mem_temp = mem; memory mem_temp2; while(mem_temp != NULL) {free(mem_temp->data); mem_temp2 = mem_temp; mem_temp = mem_temp->next; free(mem_temp2);} /*mem = NULL; mem_last = NULL; mem_size = 0.0;*/ return;}
double memory_info() {return mem_size;} // mem_size in MiB
void memory_free(void *f, size_t size) {free(f); mem_size -= ((double)(size)) / 1048576.0; return;}

// ----------------------------------------------------------------------------------------------------------------

#define AI_OP_NONE (unsigned char) 0
#define AI_OP_ADD (unsigned char) 1
#define AI_OP_SUB (unsigned char) 2
#define AI_OP_MUL (unsigned char) 3
#define AI_OP_DIV (unsigned char) 4
#define AI_OP_SQR (unsigned char) 5 // ^2 square
#define AI_OP_EXP (unsigned char) 6
#define AI_OP_TANH (unsigned char) 7
#define AI_OP_SIGMOID (unsigned char) 8
#define AI_OP_LIN (unsigned char) 9

#define new_ai_value(n) ai_value_init(n, NULL, NULL, AI_OP_NONE)

// ----------------------------------------------------------------------------------------------------------------


typedef struct ai_uint_list_node {unsigned long long data; struct ai_uint_list_node *next;} ai_uint_list_node;
typedef ai_uint_list_node *ai_uint_list;

typedef struct ai_value_node {double data; double grad; unsigned char op; struct ai_value_node *prev_1; struct ai_value_node *prev_2;} ai_value_node;
typedef ai_value_node *ai_value;

typedef struct ai_value_list_node {ai_value data; struct ai_value_list_node *next;} ai_value_list_node;
typedef ai_value_list_node *ai_value_list;

typedef struct ai_neuron_node {ai_value_list w; ai_value b; unsigned char op;} ai_neuron_node;
typedef ai_neuron_node *ai_neuron;

typedef struct ai_neuron_list_node {ai_neuron data; struct ai_neuron_list_node *next;} ai_neuron_list_node;
typedef ai_neuron_list_node *ai_neuron_list;

typedef struct ai_layer_node {ai_neuron_list neurons;} ai_layer_node;
typedef ai_layer_node *ai_layer;

typedef struct ai_layer_list_node {ai_layer data; struct ai_layer_list_node *next;} ai_layer_list_node;
typedef ai_layer_list_node *ai_layer_list;

typedef struct ai_MLP_node {ai_layer_list layers;} ai_MLP_node;
typedef ai_MLP_node *ai_MLP;

// ----------------------------------------------------------------------------------------------------------------

ai_uint_list ai_uint_list_fromArray(unsigned long long arr[], unsigned long long arrDim) {
    if(arrDim == 0) return NULL; size_t size = sizeof(ai_uint_list_node); ai_uint_list w = (ai_uint_list)memory_add(size); if(w == NULL) {return NULL;} ai_uint_list o = w; w->data = arr[0];
    unsigned long long i = 1; while(i < arrDim) {w->next = (ai_uint_list)memory_add(size); if(w->next == NULL) {return NULL;} w = w->next; w->data = arr[i]; i=i+1;} return o;}

// ----------------------------------------------------------------------------------------------------------------

ai_value ai_value_init(double data, ai_value prev_1, ai_value prev_2, unsigned char op) {
    ai_value v = (ai_value)memory_add(sizeof(ai_value_node)); if(v == NULL) return NULL; 
    v->data = data; v->prev_1 = prev_1; v->prev_2 = prev_2; v->op = op; v->grad = 0.0; return v;}
ai_value ai_value_add(ai_value x1, ai_value x2) {return ai_value_init(x1->data + x2->data, x1, x2, AI_OP_ADD);}
ai_value ai_value_sub(ai_value x1, ai_value x2) {return ai_value_init(x1->data - x2->data, x1, x2, AI_OP_SUB);}
ai_value ai_value_mul(ai_value x1, ai_value x2) {return ai_value_init(x1->data * x2->data, x1, x2, AI_OP_MUL);}
ai_value ai_value_div(ai_value x1, ai_value x2) {return ai_value_init(x1->data * x2->data, x1, x2, AI_OP_DIV);}
ai_value ai_value_sqr(ai_value x1) {return ai_value_init(x1->data * x1->data, x1, NULL, AI_OP_SQR);}
ai_value ai_value_exp(ai_value x1) {return ai_value_init(exp(x1->data), x1, NULL, AI_OP_EXP);}
ai_value ai_value_tanh(ai_value x1) {return ai_value_init(tanh(x1->data), x1, NULL, AI_OP_TANH);}
ai_value ai_value_sigmoid(ai_value x1) {return ai_value_init(sigmoid(x1->data), x1, NULL, AI_OP_SIGMOID);}
ai_value ai_value_lin(ai_value x1) {return ai_value_init(x1->data, x1, NULL, AI_OP_LIN);}

void ai_value_backward_(ai_value root) {
    if(root->op == AI_OP_ADD) {root->prev_1->grad += root->grad; root->prev_2->grad += root->grad;}
    else if(root->op == AI_OP_SUB) {root->prev_1->grad += root->grad; root->prev_2->grad += -1.0 * root->grad;}
    else if(root->op == AI_OP_MUL) {root->prev_1->grad += root->prev_2->data * root->grad; root->prev_2->grad += root->prev_1->data * root->grad;}
    else if(root->op == AI_OP_DIV) {root->prev_1->grad += (1.0 / (root->prev_2->data)) * root->grad; root->prev_2->grad += ((-1.0 * root->prev_1->data) / (root->prev_2->data * root->prev_2->data)) * root->grad;}
    else if(root->op == AI_OP_SQR) {root->prev_1->grad += 2.0 * root->prev_1->data * root->grad;}
    else if(root->op == AI_OP_EXP) {root->prev_1->grad += root->data * root->grad;}
    else if(root->op == AI_OP_TANH) {root->prev_1->grad += (1.0 - (root->data * root->data)) * root->grad;}
    else if(root->op == AI_OP_SIGMOID) {root->prev_1->grad += root->data * (1.0 - root->data) * root->grad;}
    else if(root->op == AI_OP_LIN) {root->prev_1->grad += root->grad;}
    else return;
    if(root->prev_1 != NULL) ai_value_backward_(root->prev_1); if(root->prev_2 != NULL) ai_value_backward_(root->prev_2); return;}
void ai_value_grad_reset(ai_value root) {root->grad = 0.0; if(root->prev_1 != NULL) ai_value_grad_reset(root->prev_1); if(root->prev_2 != NULL) ai_value_grad_reset(root->prev_2); return;}
void ai_value_backward(ai_value root) {ai_value_grad_reset(root); root->grad = 1.0; ai_value_backward_(root); return;}

ai_value_list ai_value_list_fromArray(double arr[], unsigned long long arrDim) {
    if(arrDim == 0) return NULL; size_t size = sizeof(ai_value_list_node);
    ai_value_list w = (ai_value_list)memory_add(size); if(w == NULL) {return NULL;}
    ai_value_list o = w; w->data = new_ai_value(arr[0]); if(w->data == NULL) {return NULL;}
    unsigned long long i = 1; while(i < arrDim) {
        w->next = (ai_value_list)memory_add(size); if(w->next == NULL) {return NULL;}
        w = w->next; w->data = new_ai_value(arr[i]); if(w->data == NULL) {return NULL;} i=i+1;}
    return o;}


// ----------------------------------------------------------------------------------------------------------------

ai_neuron ai_neuron_init(unsigned long long nin, unsigned char op) {
    if(nin == 0) return NULL; double from; double to;
    if(op == AI_OP_TANH) {from = -1.0; to = 1.0;} else if(op == AI_OP_SIGMOID) {from = 0.0; to = 1.0;} else if(op == AI_OP_LIN) {from = -1.0; to = 1.0;} else return NULL;
    size_t size = sizeof(ai_value_list_node); ai_value_list w = (ai_value_list)memory_add(size); if(w == NULL) {return NULL;} ai_value_list ww = w; w->data = new_ai_value(rand_double(from, to));
    nin = nin - 1; while(nin > 0) {w->next = (ai_value_list)memory_add(size); if(w->next == NULL) {return NULL;} w = w->next; w->data = new_ai_value(rand_double(from, to)); if(w->data == NULL) {return NULL;} nin = nin - 1;}
    ai_neuron v = (ai_neuron)memory_add(sizeof(ai_neuron_node)); if(v == NULL) return NULL; v->w = ww; v->b = new_ai_value(rand_double(from, to)); if(v->b == NULL) {return NULL;} v->op = op; return v;}

ai_value ai_neuron_call(ai_neuron n, ai_value_list x) {
    ai_value_list w = n->w; ai_value act = new_ai_value(0.0); while(w != NULL && x != NULL) {act = ai_value_add(act, ai_value_mul(w->data, x->data)); w = w->next; x = x->next;} act = ai_value_add(act, n->b);
    if(n->op == AI_OP_TANH) return ai_value_tanh(act); else if(n->op == AI_OP_SIGMOID) return ai_value_sigmoid(act); else if(n->op == AI_OP_LIN) return ai_value_lin(act); else return new_ai_value(0.0);}

// ----------------------------------------------------------------------------------------------------------------

ai_layer ai_layer_init(unsigned long long nin, unsigned long long nout, unsigned char op) { // nin = number of inputs in the layer and nout = number of outputs (number of neurons)
    if(nin == 0 || nout == 0) return NULL; size_t size = sizeof(ai_neuron_list_node); ai_neuron_list n = (ai_neuron_list)memory_add(size); if(n == NULL) {return NULL;} ai_neuron_list nn = n; n->data = ai_neuron_init(nin, op);
    nout = nout - 1; while(nout > 0) {n->next = (ai_neuron_list)memory_add(size); if(n->next == NULL) {return NULL;} n = n->next; n->data = ai_neuron_init(nin, op); if(n->data == NULL) {return NULL;} nout = nout - 1;}
    ai_layer v = (ai_layer)memory_add(sizeof(ai_layer_node)); if(v == NULL) return NULL; v->neurons = nn; return v;}

ai_value_list ai_layer_call(ai_layer l, ai_value_list x) {
    ai_neuron_list n = l->neurons; if(n == NULL) return NULL; size_t size = sizeof(ai_value_list_node); ai_value_list out = (ai_value_list)memory_add(size); if(out == NULL) {return NULL;} ai_value_list outs = out;
    while(n != NULL) {out->data = ai_neuron_call(n->data, x); if(out->data == NULL) {return NULL;} if(n->next != NULL) {out->next = (ai_value_list)memory_add(size); if(out == NULL) {return NULL;}} out = out->next; n = n->next;}
    return outs;}

// ----------------------------------------------------------------------------------------------------------------

ai_MLP ai_MLP_init(unsigned long long nin, ai_uint_list nouts, unsigned char op) { // nin = number of inputs in the network and nout = number of outputs of each layer (list)
    if(nin == 0 || nouts == NULL) return NULL; ai_uint_list n = (ai_uint_list)memory_add(sizeof(ai_uint_list_node)); if(n == NULL) {return NULL;} n->data = nin; n->next = nouts; if(n == NULL || n->next == NULL) return NULL; // not necessary (already checked before)
    size_t size = sizeof(ai_layer_list_node); ai_layer_list layers = (ai_layer_list)memory_add(size); if(layers == NULL) {return NULL;} ai_layer_list l = layers;
    while(n != NULL && n->next != NULL) {l->data = ai_layer_init(n->data, n->next->data, op); n = n->next; if(n != NULL && n->next != NULL) {l->next = (ai_layer_list)memory_add(size); if(l->next == NULL) {return NULL;}} l = l->next;}
    ai_MLP mlp = (ai_MLP)memory_add(sizeof(ai_MLP_node)); if(mlp == NULL) {return NULL;} mlp->layers = layers; return mlp;}

ai_value_list ai_MLP_call(ai_MLP mlp, ai_value_list x) {ai_layer_list l = mlp->layers; while(l != NULL) {x = ai_layer_call(l->data, x); l = l->next;} return x;}

double ai_MLP_train(ai_MLP mlp, ai_value_list xs_value[], ai_value_list ys_value, unsigned long long ai_samples, unsigned long long end, double step) {
    ai_value_list ypred; ai_value_list yp; ai_value_list ysv; ai_value_list a; ai_neuron_list n; ai_value_list w; ai_layer_list l; ai_value loss;
    double dec = 1.0 - (1.0/((double)end)); double loss_prec = 1.0; unsigned long long eq = 0;
    unsigned long long i; size_t size_value_list_node = sizeof(ai_value_list_node);
    while(end > 0) {
        // calculate ypred for each sample
        ypred = (ai_value_list)memory_add(size_value_list_node); if(ypred == NULL) {memory_free_all(); return 0;}
        yp = ypred; a = ai_MLP_call(mlp, xs_value[0]); yp->data = a->data;
        for(i=1; i<ai_samples; i++) {yp->next = (ai_value_list)memory_add(size_value_list_node); if(yp->next == NULL) {memory_free_all(); return 0;} yp = yp->next; a = ai_MLP_call(mlp, xs_value[i]); yp->data = a->data;}
        // calculate the total loss
        loss = new_ai_value(0.0); ysv = ys_value; while(ysv != NULL && ypred != NULL) {loss = ai_value_add(loss, ai_value_sqr(ai_value_sub(ysv->data, ypred->data))); ysv = ysv->next; ypred = ypred->next;}
        // filter the training
        if((loss_prec < 0.3 && loss->data >= 0.3) || step == 0.0 || eq > 1000) {end = 1;}
        else {
            if(loss->data <= 1.000001*loss_prec && loss->data >= 0.999999*loss_prec) eq += 1; else eq = 0;
            loss_prec = loss->data;
            // reset all grad = 0.0 and backpropagation on the loss
            ai_value_backward(loss);
            // update weights and biases of the MLP
            l = mlp->layers; while(l != NULL) {n = l->data->neurons; while(n != NULL) {w = n->data->w; while(w != NULL) {w->data->data = w->data->data - (step * w->data->grad); w = w->next;} n->data->b->data = n->data->b->data - (step * n->data->b->grad); n = n->next;} l = l->next;}
            // decrement the gradient decent step for the next iteration
            step = step * dec; if(step < 0.0) step = 0.0;}
        end = end - 1;}
    return loss_prec;}

// ----------------------------------------------------------------------------------------------------------------

#define AI_SAMPLES (unsigned long long) 8
#define AI_MLP_INPUTS (unsigned long long) 10
#define AI_MLP_OUTPUTS (unsigned long long) 1
#define AI_MLP_LAYERS (unsigned long long) 3 // 3 layers
#define AI_MLP_OUT_PER_LAYER {4, 4, AI_MLP_OUTPUTS} // 3 layers (1st: 4 outs (neurons), 2nd: 4 outs (neurons), 3th: 1 out (neuron))
#define AI_TRAIN_INC (double) 0.005 // decrement of weights and biases during training
#define AI_TRAIN_MAX_CYCLES 1000

// ----------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    double time_tot = ((double)clock()) / ((double)CLOCKS_PER_SEC);
    printf_color_reset();
    printf("\n\n-----------------------------------------------------------------------------\n\n");
    memory_init();
    // -----------------------------------------------------------------------------------------------------------



    double xs[AI_SAMPLES][AI_MLP_INPUTS] = {
        {2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 3.0, -1.0, 0.5, 4.5},
        {3.0, -1.0, 0.5, -1.0, 3.0, -1.0, 0.5, 1.0, 1.0, 0.0},
        {0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 2.5},
        {3.0, -1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0},
        {-1.0, 0.5, 4.5, -1.0, 3.0, -1.0, 2.0, 3.0, 0.5, 3.0},
        {0.0, 1.0, 1.0, -1.0, 3.0, 1.0, 1.0, 0.0, -1.0, 0.5},
        {1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0},
        {1.0, 0.0, -1.0, 0.5, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0}
    };

    double ys[AI_SAMPLES] = {1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0};


    // MLP init
    ai_value_list ys_value = ai_value_list_fromArray(ys, AI_SAMPLES);
    ai_value_list xs_value[AI_SAMPLES]; unsigned long long i; for(i=0; i<AI_SAMPLES; i++) {xs_value[i] = ai_value_list_fromArray(xs[i], AI_MLP_INPUTS);}
    unsigned long long louts[AI_MLP_LAYERS] = AI_MLP_OUT_PER_LAYER;
    ai_MLP mlp = ai_MLP_init(AI_MLP_INPUTS, ai_uint_list_fromArray(louts, AI_MLP_LAYERS), AI_OP_TANH);
    
    // TRAINING
    double loss = ai_MLP_train(mlp, xs_value, ys_value, AI_SAMPLES, AI_TRAIN_MAX_CYCLES, AI_TRAIN_INC);

    // OUTPUT RESULT (MLP TRAINED HERE)
    ai_value_list result; unsigned long long s;
    printf("Loss = %.6f\n", loss); for(s=0; s<AI_SAMPLES; s++) {result = ai_MLP_call(mlp, ai_value_list_fromArray(xs[s], AI_MLP_INPUTS)); printf("%llu° sample :\t ys = %.3f | ypred = %.3f\n", s+1, ys[s], result->data->data);}

    // OUTPUT NEURAL NET INFO
    unsigned long long n_weights = AI_MLP_INPUTS * louts[0]; unsigned long long n_biases = 0; for(s=0; s<AI_MLP_LAYERS-1; s++) {n_weights += louts[s] * louts[s+1]; n_biases += louts[s];} n_biases += louts[s];
    unsigned long long params = n_weights + n_biases; 
    printf("\n@ Neural Net :\nParameters = %llu (%llu weights + %llu biases)\nNeurons = %llu\nInputs = %llu\nOutputs = %llu\nLayers = %llu\nTraining samples = %llu\nTraining inc. = %.6f", params, n_weights, n_biases, n_biases, AI_MLP_INPUTS, AI_MLP_OUTPUTS, AI_MLP_LAYERS, AI_SAMPLES, AI_TRAIN_INC);
    
    


    
    // -----------------------------------------------------------------------------------------------------------
    memory_free_all();
    time_tot = (((double)clock()) / ((double)CLOCKS_PER_SEC)) - time_tot; // seconds
    
    printf_color_red();
    printf("\n\n-----------------------------------------------------------------------------\n@ Memory:\t%.3f MiB (%.3f MB)\t\t\t%.0f kiB (%.0f kB)\n@ Time:\t\t%.3f s\t\t\t\t\t%.0f ms\n-----------------------------------------------------------------------------\n", memory_info(), memory_info() * 1.048576, memory_info() * 1024.0, memory_info() * 1048.576, time_tot, time_tot*1000.0);
    printf_color_reset();
    return 0;
}

