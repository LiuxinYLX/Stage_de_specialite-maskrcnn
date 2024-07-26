# Connexion au serveur

## Créer la clé personnelle

Créer trois fichiers : id_rsaIMB, id_rsaIMB.pub et config

Dans le fichier config : 

Host ssh.plafrim.fr
	IdentityFile ~/.ssh/id_rsaIMB
	IdentitiesOnly yes

Host plafrim
     ForwardAgent yes
     ForwardX11 yes
     User lyang
     ProxyCommand ssh -A -l lyang ssh.plafrim.fr -W plafrim:22



## Rendre le fichier accessible uniquement par le propriétaire 

chmod 600 ~/.ssh/id_rsaIMB



## Ajouter la clé privée à l'agent SSH 

ssh-add ~/.ssh/id_rsaIMB

Pour vérifier : ssh-add -l



## Se connecter au plateforme

ssh lyang@plafrim

Nous pouvons entrer dans le dossier beegfs/lyang où il y a plus de mémoire.

# Utilisation

## srun

- Mode interactive : Réserver des ressources et y démarrer un job interactif

- srun [options] commande arguments

- Les ressources sont libérées à la fin de l’exécution du programme

- tests, exécutions rapides ou interactives,
  session interactive sur un ou plusieurs nœuds de calcul
  (compilations/exécutions). Attente potentielle à chaque lancement
  
  

## salloc

- Mode interactive : Réserver des ressources pour une session interactive + srun

- $> salloc [options]
  $> srun [opt1] prog1 arg1
  $> srun [opt2] prog2 arg2
  $> exit

- session de travail interactive de plusieurs heures, plusieurs programmes à
  tester



## sbatch

- Mode batch

- #!/bin/bash
  #SBATCH [options]
  #SBATCH [options]
  srun [opt1] prog1 arg1
  srun [opt2] prog2 arg2

- Campagne de calculs. Possibilité de tester/valider en amont avec salloc



## Options spécifiques

-I : immédiat ou délai max en secondes (--immediat)

--pty bash : pour entrée/sorties interactives

– Nœud GPU : -C p100 , -C v100 , -C a100



## GPU

salloc -C p100 --time=2:0:0 

ssh (nom du node)

conda activate env