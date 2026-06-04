# CNN *from scratch* vs Transfer Learning (EfficientNet-B0) — Cats vs Dogs

Projet de TP (Deep Learning, **Dakar Institute of Technology**) comparant un
**CNN entraîné from scratch** et un modèle en **transfer learning**
(EfficientNet-B0 pré-entraîné sur ImageNet) sur le jeu de données **Cats vs Dogs**.

**Auteur : Josias AHOGA**

---

## Objectif

Comparer deux approches sur le même jeu *Cats vs Dogs* et mesurer l'impact du
transfert sur la convergence, la performance et la robustesse :

- **Expérience A** : CNN construit et entraîné *from scratch* (4 blocs de
  convolution, BatchNorm + Dropout).
- **Expérience B** : transfer learning avec EfficientNet-B0 pré-entraîné, dont la
  couche finale est adaptée à 2 classes, avec fine-tuning progressif.

Les métriques (loss, accuracy, précision, recall) sont suivies à chaque époque,
sur train et validation. Deux optimiseurs sont testés (Adam et SGD), un
learning rate finder guide le choix du LR, un scheduler est utilisé, le meilleur
modèle est sauvegardé puis rechargé pour le test final, et une matrice de
confusion est tracée.

---

## Résultats obtenus

| Expérience | Stratégie | Meilleure val accuracy |
|---|---|---|
| CNN from scratch + Adam | MixUp, 25 ép. | 0.9524 |
| CNN from scratch + SGD | MixUp, 25 ép. | 0.9200 |
| EfficientNet-B0 + Adam | fine-tuning progressif | 0.9844 |
| EfficientNet-B0 + SGD | fine-tuning progressif | 0.9798 |

**Test final** (meilleur modèle rechargé, 2500 images de test) :
**accuracy = 0.9824, précision = 0.9827, recall = 0.9824**.
Matrice de confusion : 1243 chats et 1213 chiens correctement classés, pour
seulement 44 erreurs sur 2500 images.

Le projet intègre plusieurs techniques avancées : backbone EfficientNet-B0,
fine-tuning progressif (la phase 2 fait passer EfficientNet+Adam de 0.9711 à
0.9844), MixUp, early stopping, et journalisation TensorBoard. L'analyse
complète figure dans le notebook (section 9).

## Environnement

Le projet a été exécuté sur **Google Colab avec GPU T4** (Internet activé pour télécharger les poids EfficientNet-B0 pré-entraînés).

Pour installer les dépendances en local :

```bash
pip install -r requirements.txt
```

```bash
pip install torch torchvision scikit-learn matplotlib pandas
```

Bibliothèques principales : PyTorch, torchvision, scikit-learn, matplotlib,
pandas. Le code détecte automatiquement le GPU (CUDA) ; à défaut il tourne sur
CPU (beaucoup plus lent).

---

## Données

Jeu *Cats vs Dogs*, déjà séparé en `train/` et `test/`, chaque dossier
contenant les sous-dossiers `cat/` et `dog/` :

```
Cat_Dog_data/
├─ train/
│  ├─ cat/   (11250 images)
│  └─ dog/   (11250 images)
└─ test/
   ├─ cat/   (1250 images)
   └─ dog/   (1250 images)
```

Les données proviennent du jeu **Cats vs Dogs de Kaggle**. Elles **ne sont pas
versionnées** dans ce dépôt (volumineuses) et se téléchargent séparément ici :

**[Télécharger les données (Google Drive)](https://drive.google.com/file/d/1SnQjreIX6y7wiYqdwEgfY7IiVJIo10X0/view?usp=sharing)**

### Mise en place des données

**Sur Google Colab (méthode utilisée dans ce projet) :**

1. Télécharger l'archive `.zip` depuis le lien ci-dessus.
2. **Déposer l'archive `.zip` (sans la décompresser) dans son propre Google
   Drive**, par exemple dans un dossier `TP_CNN/`.
3. Adapter la variable `DRIVE` au début du notebook pour pointer vers ce dossier.
   Le notebook se charge ensuite de copier l'archive sur le disque local de la
   session Colab puis de la décompresser automatiquement (la lecture locale est
   bien plus rapide que la lecture directe depuis le Drive).

**En local :** décompresser l'archive de façon à obtenir l'arborescence ci-dessus,
puis faire pointer la variable `DATA_DIR` vers le dossier `Cat_Dog_data`.

Un split train/validation (20 %, stratifié et reproductible) est réalisé
automatiquement dans le dossier `train/` ; le dossier `test/` n'est utilisé que
pour l'évaluation finale.

---

## Utilisation

Le projet tient dans un seul notebook : **`TP_CatsVsDogs_JosiasAHOGA.ipynb`**.

1. Ouvrir le notebook dans Google Colab.
2. Activer le GPU : `Exécution` → `Modifier le type d'exécution` → `T4 GPU`.
3. Adapter le chemin des données (variable `DRIVE` / `DATA_DIR`) au début du
   notebook.
4. Exécuter les cellules dans l'ordre (`Exécution` → `Tout exécuter`).

Le notebook est organisé en deux parties :

- **Partie 1 — Données** : chargement avec `ImageFolder`, transformations
  (avec justification), augmentation de données, split train/validation,
  visualisation.
- **Partie 2 — TP** : fonctions d'entraînement/évaluation, LR finder,
  CNN from scratch (Adam + SGD), transfer learning EfficientNet-B0 (Adam + SGD),
  comparaison, test final, matrice de confusion et analyse.

---

## Choix techniques

- **EfficientNet-B0** : backbone moderne, bon compromis précision / paramètres.
- **Fine-tuning progressif** : tête d'abord (backbone gelé), puis dégel des dernières couches avec un LR faible.
- **MixUp** : mélange d'images et d'étiquettes pour régulariser le from scratch.
- **Early stopping** : arrêt quand la validation stagne.
- **TensorBoard** : journalisation interactive des métriques.
- **BatchNorm** après chaque convolution : stabilise et accélère l'apprentissage.
- **Dropout** dans la tête dense : limite le surapprentissage.
- **Augmentation de données** (rotation, recadrage, flip) sur l'entraînement
  uniquement, pour améliorer la généralisation ; validation/test sans
  augmentation pour une évaluation honnête.
- **Normalisation ImageNet** : nécessaire pour exploiter le backbone pré-entraîné.
- **Scheduler** (cosine / step) : décroissance du learning rate.
- **Seed fixé** : reproductibilité des expériences.
- **Sauvegarde du meilleur modèle** (`.pth`) selon la val accuracy, puis
  rechargement pour le test final. Les fichiers `.pth` ne sont pas versionnés.

---

## Structure du dépôt

```
.
├─ TP_CatsVsDogs_JosiasAHOGA.ipynb   # notebook complet (cours + TP)
├─ README.md
└─ .gitignore                        # exclut données, modèles, caches
```

---

## Reproductibilité

Toutes les expériences utilisent un seed fixé (`SEED = 42`). En réexécutant le
notebook de bout en bout sur un GPU, on retrouve des résultats équivalents à
ceux rapportés ci-dessus (de légères variations sont possibles selon la version
de CUDA / PyTorch).

---

## Licence

MIT.
