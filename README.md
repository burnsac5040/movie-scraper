## Web Scraping with Python
---------------------------

### [Kelley Blue Book](kbb)
--------------------
This was my first serious project that I uploaded to Github. It has been moved to this directory to make my projects look a little bit cleaner. I used BeautifulSoup to scrape KBB and then cleaned the data. I thought that I was going to be able to do more with it than what I was. I think I will come back to this at some point.


### [2019 Movie Scraping from Wikipedia](wikimovies)
--------------------------------------
This is a project where I scraped and cleaned data from a Wikipedia table containing information about 2019 films.

I then did some graphing of the data and attempted to perform linear regression on the data with PyTorch where I was using the budget to predict the box office performance.  The R2 score was 64%, however the loss was extremely high.


### [Youtube Transcripts](youtube-transcripts)
An attempt to compare liberal vs conservative YouTube channels' video titles and transcripts.  This is a WIP and something that I had started a while ago and need to finish.  I used [`youtube-channel-transcript-api`](https://pypi.org/project/youtube-channel-transcript-api/) and [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/) to get video information.

The reason why I stopped the project at the point I did was because the process of scraping each channel was taking a very long time, especially for the channels that have 10,000 videos.


### [Linux Distributions](distrowatch)
-----------------------
I scraped the top 275 distributions on [Distrowatch](https://distrowatch.com/) and got the basic specs (e.g., distribution it is based on, architecture, type of desktop) and the community rating for each of them. Then I created a dataframe that has the features of every version of the distribution (there is a .csv containing this). For this project I only used the most recent release.

I 'one-hot-encoded' the data and then classified it based on its' rating to perform a logistic regression. The model did better than chance, however the data that I used may not be the best.  This was just a little project I did because I just recently got into Linux.

Overall, I would have to say that (with the data I have) the specs of a distribution does not really correlate with what people think of it.


### [Trulia House Prices](trulia)
-----------------------
Scraped 2500 house prices from 5 cities in Missouri. I then cleaned the data and created a few graphs.  The R2 score from the linear regression performed was 0.72, however the MSE was huge. A better selection of the data needs to be done to work with a linear regression model.
