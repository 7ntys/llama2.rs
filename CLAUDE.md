# Contexte : projet llama2.c → Rust de Julien

## Qui est Julien
- Jeune diplômé ingénieur, arrive chez AWS Dublin (Systems Engineer L4, managed ops, Python/TypeScript au travail).
- Dev iOS solo (Swift), deux apps publiées.
- Objectif long terme : inference engineering dans les grands labs d'IA. Veut à terme contribuer à
  mistral.rs, vllm-router et NVIDIA Dynamo.
- Niveau de départ : quasiment aucun background en ML. Connaît la syntaxe de base de Rust.
  N'a jamais fait de C.

## Objectif du projet
- Porter `run.c` de karpathy/llama2.c en Rust, from scratch (CPU, sans framework ML).
- À terme, le faire évoluer en serveur d'inférence : KV cache paginé, continuous batching,
  prefix caching, routeur KV-aware.
- Veut aussi le valoriser sur son CV rapidement. Viser un MVP rapide mais solide :
  il doit pouvoir expliquer chaque ligne en entretien.
- Formulation CV : « from-scratch LLM inference engine in Rust (Llama architecture) »,
  pas « recreating GPT » (ce n'est ni GPT, ni de l'entraînement).

## Règles pédagogiques (importantes)
- **Ne pas écrire le code à sa place.** Expliquer les concepts, le C de run.c, les erreurs du
  borrow checker, les divergences de résultats. Donner au plus des signatures ou des indices.
- Il ne doit pas regarder les ports Rust existants de llama2.c avant d'avoir fini le sien.
- Répondre en français.

## Ce qui marche pour lui
- Explications **très simples**, avec des **exemples numériques minuscules** calculables à la main
  (vecteurs de 2 à 4 nombres, 2 têtes, 3 tokens).
- Des analogies :
  - Attention = une réunion : query = la question qu'on pose, key = le badge qu'on porte,
    value = ce qu'on raconte si on vient nous voir ; softmax = répartir son attention en % ;
    sortie = les notes prises. Variante : demander conseil à des amis pour un restaurant.
  - Feed-forward = retour au bureau, chacun réfléchit seul ; neurones = détecteurs.
  - Connexion résiduelle = agrafer ses notes sur sa fiche au lieu de la jeter.
  - Multi-head = poser plusieurs questions différentes en parallèle.
- Une étape à la fois. Les explications denses avec des formes de tenseurs d'emblée, ou plusieurs
  concepts d'un coup, ne passent pas. Il le dit franchement quand il n'a pas compris : reformuler
  plus simplement, pas plus longuement.

## Ce qu'il a vu
- Vidéo de Karpathy « Let's build GPT » (en grande partie floue, notamment l'entraînement) et
  le blog « GPT in 60 Lines of NumPy » de Jay Mody (picoGPT), qui l'a beaucoup aidé.
- Concepts compris à un niveau correct : tokens/vocabulaire, embedding table (lookup d'une ligne),
  positional embedding (addition case par case), B/T/C et le rangement plat en mémoire
  (index = b*T*C + t*C + c), boucle de génération (forward → logits du dernier token → softmax →
  sampling → append), attention (q·k, /√d, masque causal, softmax, somme pondérée des v),
  multi-head (couper q/k/v en tranches par tête, recoller), feed-forward (élargir → non-linéarité
  → réduire), bloc pre-norm avec résiduelles, empilement de N blocs, KV cache (idée générale).
- Encore fragile : la vision d'ensemble, LayerNorm/RMSNorm, les manipulations de formes.
- Pas encore vu : RoPE, SwiGLU, GQA (sauf de nom), le détail du KV cache dans run.c,
  le format du checkpoint, le tokenizer BPE.

## État actuel
- llama2.c cloné et compilé. `./run stories15M.bin` fonctionne : ~218 tok/s avec `make run`
  (baseline à noter ; tester aussi `make runfast`).
- À faire : enregistrer la sortie de référence déterministe `./run stories15M.bin -t 0`.
  C'est le test final : le port Rust doit produire exactement ce texte.
- `rmsnorm` déjà expliquée (exemple : x=[3,4], weight=[1,2] → [0.849, 2.263]).

## Plan (ordre de construction)
1. Survoler la structure de run.c, lire `generate` en premier pour la vue d'ensemble.
2. `cargo new` + lire la Config (7 int32 en tête de fichier ; vocab_size négatif = poids du
   classifieur non partagés avec l'embedding). Afficher et comparer avec le C.
3. Charger les poids en suivant `memory_map_weights` : un Vec<f32> par tenseur, lecture via
   chunks_exact(4) + f32::from_le_bytes. Pas de mmap/unsafe au début.
4. rmsnorm, softmax, matmul, chacun avec un test sur un petit exemple fait à la main.
5. forward (expliquer RoPE, GQA, KV cache, SwiGLU au moment où il les rencontre).
6. Sampler argmax + température.
7. Décodage des tokens → **MVP** = sortie identique au C avec -t 0.
8. Reportés après le MVP : encodage BPE du prompt, top-p.
9. Ensuite : bench vs C, rayon sur matmul, quantization int8 (runq.c comme oracle),
   modèles 42M/110M.

Validation : en cas de divergence, ajouter des printf dans le C pour dumper les activations
couche par couche et comparer avec une tolérance de ~1e-5.

## Pièges Rust à anticiper
- Toujours `cargo run --release` (debug = 10 à 50 fois plus lent).
- Fonctions de calcul sur des slices : `fn matmul(out: &mut [f32], x: &[f32], w: &[f32], n, d)`.
- run.c appelle `rmsnorm(x, x, ...)` (même buffer en entrée et en sortie) : impossible tel quel
  en Rust → variante en place ou buffer temporaire.
- Zéro allocation dans la boucle de génération (tous les buffers dans un RunState).
- Si Rc<RefCell> ou des .clone() partout apparaissent, revoir les signatures.

## Feuille de route après le port
KV cache paginé → continuous batching + chunked prefill → prefix caching → routeur
multi-instances KV-aware → éventuellement désagrégation prefill/decode. Avec un harnais de bench
(TTFT, ITL, débit, goodput). Lectures associées : PagedAttention (vLLM), Orca, Sarathi-Serve,
SGLang/RadixAttention, DistServe.
