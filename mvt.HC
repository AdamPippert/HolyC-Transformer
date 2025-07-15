//-----------------------------
// Minimum Viable Transformer
// Adam Pippert · 2025‑07‑15
// Pure HolyC (compile with hcc)
//-----------------------------
#define D_MODEL     16
#define D_FF        32
#define VOCAB       27      // a‑z + space
#define MAX_SEQ     32

// Simple RNG (LCG) for deterministic demo
I64 rnd_state = 1;
F64 Rand01() {
    rnd_state = (rnd_state*48271) % 0x7fffffff;
    return ToF64(rnd_state) / 2147483647.0;
}

// Embedding & weight tensors
F64 Embed[VOCAB][D_MODEL];
F64 W_Q[D_MODEL][D_MODEL], W_K[D_MODEL][D_MODEL], W_V[D_MODEL][D_MODEL];
F64 W1[D_MODEL][D_FF],    W2[D_FF][D_MODEL];
F64 W_OUT[D_MODEL][VOCAB];

// He‑style init
U0 InitMat(F64 *m, I64 rows, I64 cols) {
    F64 scale = 1.0 / sqrt(ToF64(rows));
    for (I64 i=0;i<rows*cols;++i) m[i] = (Rand01()*2.0-1.0)*scale;
}

// Soft‑max in‑place on vector `v` length n
U0 Softmax(F64 *v, I64 n) {
    F64 max = v[0];
    for (I64 i=1;i<n;++i) if (v[i]>max) max=v[i];
    F64 sum=0;
    for (I64 i=0;i<n;++i) { v[i]=2.718281828 ` (v[i]-max); sum+=v[i]; }
    for (I64 i=0;i<n;++i) v[i]/=sum;
}

// Dot‑product of len d
F64 Dot(F64 *a, F64 *b, I64 d) {
    F64 s=0; for(I64 k=0;k<d;++k) s+=a[k]*b[k]; return s;
}

// Forward one sequence (length n)
// tokens[]  : I64 ids
// out_probs : probability dist for next char
U0 Forward(I64 *tokens, I64 n, F64 *out_probs) {
    static F64 X[MAX_SEQ][D_MODEL];
    // 1 Embedding lookup
    for(I64 i=0;i<n;++i)
        for(I64 j=0;j<D_MODEL;++j)
            X[i][j]=Embed[tokens[i]][j];

    // 2 Q,K,V
    static F64 Q[MAX_SEQ][D_MODEL],K[MAX_SEQ][D_MODEL],V[MAX_SEQ][D_MODEL];
    for(I64 i=0;i<n;++i)
      for(I64 j=0;j<D_MODEL;++j){
        F64 q=0,k=0,v=0;
        for(I64 t=0;t<D_MODEL;++t){
          F64 x=X[i][t];
          q+=x*W_Q[t][j];
          k+=x*W_K[t][j];
          v+=x*W_V[t][j];
        }
        Q[i][j]=q; K[i][j]=k; V[i][j]=v;
      }

    // 3 Attention scores & context
    static F64 Attn[MAX_SEQ][MAX_SEQ];
    static F64 Z[MAX_SEQ][D_MODEL];
    F64 scale=1.0/sqrt(ToF64(D_MODEL));
    for(I64 i=0;i<n;++i){
        // scores
        for(I64 j=0;j<n;++j){
            Attn[i][j]=scale*Dot(Q[i],K[j],D_MODEL);
        }
        Softmax(Attn[i],n);          // row soft‑max
        // weighted sum
        for(I64 d=0;d<D_MODEL;++d){
            F64 s=0;
            for(I64 j=0;j<n;++j) s+=Attn[i][j]*V[j][d];
            Z[i][d]=s;
        }
    }

    // 4 Feed‑Forward per token (use last token for LM head)
    static F64 H[D_FF];
    I64 iLast=n-1;
    for(I64 j=0;j<D_FF;++j){
        F64 s=0;
        for(I64 d=0;d<D_MODEL;++d) s+=Z[iLast][d]*W1[d][j];
        H[j]= (s>0)?s:0;             // ReLU
    }
    static F64 Y[D_MODEL];
    for(I64 d=0;d<D_MODEL;++d){
        F64 s=0; for(I64 j=0;j<D_FF;++j) s+=H[j]*W2[j][d];
        Y[d]=s;
    }

    // 5 Linear → vocab + soft‑max
    for(I64 v=0;v<VOCAB;++v){
        F64 s=0; for(I64 d=0;d<D_MODEL;++d) s+=Y[d]*W_OUT[d][v];
        out_probs[v]=s;
    }
    Softmax(out_probs,VOCAB);
}

// ------------------ Demo main ------------------
U0 Main() {
    // Initialise weights
    InitMat(&Embed[0][0], VOCAB, D_MODEL);
    InitMat(&W_Q[0][0],  D_MODEL, D_MODEL);
    InitMat(&W_K[0][0],  D_MODEL, D_MODEL);
    InitMat(&W_V[0][0],  D_MODEL, D_MODEL);
    InitMat(&W1[0][0],   D_MODEL, D_FF);
    InitMat(&W2[0][0],   D_FF,    D_MODEL);
    InitMat(&W_OUT[0][0],D_MODEL, VOCAB);

    // Tokenize "hello "
    I64 seq[6];
    U8 *txt="hello ";
    for(I64 i=0;i<6;++i){
        U8 c=txt[i];
        seq[i]=(c==' ')?26:(c-'a');
    }

    F64 probs[VOCAB];
    Forward(seq,6,probs);

    "Next‑char probs for \"hello \":\n";
    for(I64 v=0;v<VOCAB;++v){
        "%c : %.3lf\n",(v==26)?' ':('a'+v),probs[v];
    }
}
Main;
