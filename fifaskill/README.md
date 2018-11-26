# trueskill_augmented
An Augmentation for the Trueskill Video Game rater with attention to offense/defense 
and player contribution variables


## Requirements
Edward  
Tensorflow  

## Data
To solve the player/team evaluation problem we use a kaggle dataset on 
European Soccer Games available [here](https://www.kaggle.com/hugomathien/soccer). 
This data set contains seven tables: countries, leagues, matches, player, player 
attributes, teams, and team attributes. We are primarily concerned with match data 
from different teams. The player and team attributes are from FIFA (the video game) 
and are unlikely to be used

## Box's Loop
### Model
We attempt to create a solution that is efficiently scalable to a large number of teams 
and players. Individual player skill is modeled as a Gaussian, with the mean as their skill 
rating and variance as therandomness in their performance. The team rating is modeled 
dependently on the player skills and theirindividual impact to the team(all imapacts summing to 1),
 ignoring team attributes given by the dataset. We will use this model to augment the existing 
TrueSkill model 

### Inference
One key change we will be making with our new model is that we will be implementing the model using 
Edward and conducting inference using Black Box Variational Inference. Guo (2011) uses Variational EM 
with exact updates

### Criticism
Our evaluation criteria will follow that of Guo (2011) which is an estimate of classification 
accuracy using the area under the curve measure of the ROC curve. In hyperparameter tuning, 
we will evaluate theimpact of our player-based scoring model by comparing the AUC of our model 
and the the original


## References
[Shengbo Guo: Bayesian Recommendar Systems](http://users.cecs.anu.edu.au/~sguo/thesis.pdf)  
[Ralf Herbrich, Tom Minka, Thore Graepel: Trueskill](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/)  
[Data](https://www.kaggle.com/hugomathien/soccer).
