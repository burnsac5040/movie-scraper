## 2019 Movie Scraping from Wikipedia
This is a project where I scraped and cleaned data from a Wikipedia table containing information about 2019 films.  

I then did some graphing of the data and attempted to perform linear regression on the data with PyTorch where I was using the budget to predict the box office performance.  The R2 score was 64%, however the loss was extremely high.

## Linux Distributions
I scraped the top 275 distributions on [Distrowatch](https://distrowatch.com/) and got the basic specs (e.g., distribution it is based on, architecture, type of desktop) and the community rating for each of them. Then I created a dataframe that has the features of every version of the distribution (there is a .csv containing this). For this project I only used the most recent release.

I 'one-hot-encoded' the data and then classified it based on its' rating to perform a logistic regression. The model did better than chance, however the data that I used may not be the best.  This was just a little project I did because I just recently got into Linux. 

Overall, I would have to say that (with the data I have) the specs of a distribution does not really correlate with what people think of it.
