# CNN *from scratch* vs Transfer Learning — Cats vs Dogs

Projet de TP (Deep Learning, **Dakar Institute of Technology**) comparant un
**CNN entraîné from scratch** et un modèle en **transfer learning** (ResNet18
pré-entraîné sur ImageNet) sur le jeu de données **Cats vs Dogs**.

**Auteur : Josias AHOGA**

---

## Objectif

Comparer deux approches sur le même jeu *Cats vs Dogs* et mesurer l'impact du
transfert sur la convergence, la performance et la robustesse :

- **Expérience A** : CNN construit et entraîné *from scratch* (4 blocs de
  convolution, BatchNorm + Dropout).
- **Expérience B** : transfer learning avec ResNet18 pré-entraîné, dont la
  couche finale est adaptée à 2 classes.

Les métriques (loss, accuracy, précision, recall) sont suivies à chaque époque,
sur train et validation. Deux optimiseurs sont testés (Adam et SGD), un
learning rate finder guide le choix du LR, un scheduler est utilisé, le meilleur
modèle est sauvegardé puis rechargé pour le test final, et une matrice de
confusion est tracée.

---

## Résultats obtenus

| Expérience | Époques | Meilleure val accuracy |
|---|---|---|
| CNN from scratch + Adam | 25 | 0.9353 |
| CNN from scratch + SGD | 25 | 0.7993 |
| Transfer learning (ResNet18) + Adam | 10 | 0.9780 |
| Transfer learning (ResNet18) + SGD | 10 | 0.9782 |

**Test final** (meilleur modèle rechargé, 2500 images de test) :
**accuracy = précision = recall = 0.9784**.
Matrice de confusion : 1226 chats et 1220 chiens correctement classés, pour
seulement 54 erreurs sur 2500 images.

Principaux enseignements : le transfer learning atteint ~0.97 dès la première
époque (contre 25 époques pour que le from scratch approche 0.93) ; pour le CNN
from scratch, Adam (0.935) surpasse largement SGD (0.799), alors que pour le
transfer les deux optimiseurs sont équivalents. L'analyse complète figure dans
le notebook (section 2.7).

---

## Environnement

Le projet a été exécuté sur **Google Colab avec GPU T4**.

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
  CNN from scratch (Adam + SGD), transfer learning ResNet18 (Adam + SGD),
  comparaison, test final, matrice de confusion et analyse.

---

## Choix techniques

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
