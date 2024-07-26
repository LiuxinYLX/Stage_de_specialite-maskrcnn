

# Training on Your Own Dataset

Commencez par lire ce post de blog : [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). Il couvre le processus complet de création et de préparation d’un jeu de données à partir de zéro pour l’entraînement d’un modèle Mask R-CNN.


## Installation
1. Clonez le dépôt **Mask_RCNN** à partir du site [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
2. Clonez ce dépôt **nematode** et placez-le dans le répertoire **samples/**.
3. Installez les dépendances :
   ```bash
   pip3 install -r requirements.txt
   ```
4. Exécutez l'installation depuis le répertoire **Mask_RCNN**

    ```bash
    python3 setup.py install
    ``` 
5. Téléchargez les poids pré-entraînés de COCO (mask_rcnn_coco.h5) depuis [releases page](https://github.com/matterport/Mask_RCNN/releases).

6. Lancez l'entraînement avec cette commande depuis le répertoire nematode. Nous spécifions que l'entraînement doit commencer à partir des poids pré-entraînés sur le jeu de données COCO.
   ```python
   python3 nematode.py train --dataset=nematode_dataset/ --model=coco
   ```
   Pour reprendre l'entraînement en cas d'interruption :

   ```python
   python3 nematode.py train --dataset=nematode_dataset/ --model=last
   ```
