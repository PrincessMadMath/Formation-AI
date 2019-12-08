## Objectifs

Le but de ce workshop est: 
- Expérience avec ML.net
- Comprendre la tâche de classification
- Se familiariser avec le starter C# de BomberJam


## Mise en contexte

 Dans le produit Apricot on s'intéresse au lifecycle des groupes O365, par exemple on veut savoir lorsqu'un groupe devient
 inutilisé pour offrir aux utilisateurs d'archiver ce groupe. On veut créer un modèle qui est capable de prédire si un 
 groupe est "actif" ou "inactif" basé sur les données qu'on a de disponible. 
 
 Bien qu'il s'agit d'un problème de classification binaire (car il y a seulement 2 possibilité), pour atteindre l'objectif
 de mieux se préparer au BomberJam nous allons utiliser les algorithmes et les métriques de Classification multi-classe.2
 
 Pour entrainer notre modèle nous avons anonymisé et exporté du data de la prod d'Apricot (le data disponible est montré
 dans la classe `ActivityData`). Bien sur comme la classification est une tâche supervisé le data fournit est labeled: 
 il contient donc si pour la vérité qu'on recherche: si un groupe est actif ou inactif. 
 

## Étapes à suivre

1. Télécharger le data sur votre poste (va être disponible dans le channel slack)(mettre à jour variable dans Program.cs)
2. Déterminer quel features utiliser (quel information est utile) (la fonction ExtractFeatures dans RawSmartBot)
3. Évaluer votre modèle à l'aide des différentes métriques de disponible
4. Valider l'importance de vos features avec le PFI
5. Faire l'examen pour s'assurer que votre modèle supporte certains edge cases.


TLDR: Suivre les commentaires "TODO-{number}".