## March Madness Bracket Generator

### Sumedh Garimella

### About

This is the repository containing the machine learning code I've used to build my brackets for March Madness since 2022. My general process every year is to build a training dataset of match data from previous years, the prediction set consisting of the bracket for the current year, and then run an ensemble of models to determine the best model every day until the tournament starts. I predict scores for each match, including the First Four, and I've also made women's brackets, although I don't spend as much time on that as I probably should. 

### Improvements

- I'm thinking of making some Claude skills for this year's tournament to maybe speed up my data collection process.
- I'd also like to make a basic website to allow others to build their own bracket models this year! 
- Probably going to add more categories of data, dynamic features, and maybe some additional model types. Maybe I'll also try to experiment between loss/evaluation functions.