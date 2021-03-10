## Web Scraping with Python
---------------------------

### [Kelley Blue Book](kbb)
NOTE: First serious project I uploaded to GitHub. It had its own repository and was moved here to make my projects a little bit cleaner.

- [Scraping](kbb/kbb_scraper.py): `BeautifulSoup` was used to scrape [Kelley Blue Book](https://www.kbb.com) for vehicle data within a 75 mile radius of the area that I live in. Information that was scraped includes vehicle title, price, mileage, whether the vehicle was in an accident, whether it had a previous owner, interior and exterior color, etc.

- [Cleaning](kbb/kbb_analysis.py): The data was cleaned as best as I could at the time and at first I thought that I would be able to do more with the data than what I actually did. This is sometihng that I could come back to and finish the analysis a litte bit more than what I did.

--------------------------------------
### [2019 Movie Scraping from Wikipedia](wikimovies)

- [Project](wikimovies/movie-scraper.py): `BeautifulSoup` was used to scrape United States films made in 2019 from [Wikipedia](https://en.wikipedia.org/wiki/List_of_American_films_of_2019). All of the data located in the box under the movie title on Wikipeia was scraped. The data in this project was much easier to perform exploratory data analysis once it was cleaned and uniform.

- At the end of the file I attempted to perform a linear regression with `pytorch` where the budget was being used to predict the box office perforance. The R2 score was 64%, however the loss was extremely high. Because of this, I tried to turn the problem into a classification problem where I performed a logistic regression.

----------------------
### [YouTube Transcripts](youtube-transcripts)

- [Project](youtube-transcripts/yt-scraper.py): [`youtube-channel-transcript-api`](https://pypi.org/project/youtube-channel-transcript-api/) and [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/) were used to get YouTube video information. The idea was to compare liberal vs conservative YouTube channels' video titles and transcripts.

- This is another project that would need to be picked back up later as it was left like this because the time it was taking to scrape some of these channels was taking an extremely long amount of time. For instance, TheYoungTurks had around 40,000 videos to scrape data from.

-----------------------
### [Linux Distributions](distrowatch)

- [Scraping](distrowatch/scrapen-wiki.py): `pandas` was used to scrape the top 275 Linux distributions on [Distrowatch](https://distrowatch.com/). Each operating system has a table containing its specs (e.g., distribution the operating system is based on, its architecture, type of desktop, etc.) and the community rating (popularity).

- [Cleaning](distrowatch/scrapenclean.py): There is a [csv](distrowatch/data/df-os_info.csv) containing a generic overview of each operating system, as well as a [csv](distrowatch/data/df-all_versions.csv) that contains a lot more information about every release of every distribution.  The last [csv](distrowatch/data/df-ohe.csv) that may be useful was cleaned and put in a one-hot-encoded format.

- [Analysis](distrowatch/logreg.py): A logistic algorithm coded from scratch was used to analyze the data.  The model did better than chance, however; the data that was used may not have been the best to perform such a task--or, perhaps there is no correlation between the operating system's specs and its' community rating.

-----------------------
### [Trulia House Prices](trulia)

- [Scraping](trulia/trulia_scraper.py): `BeautifulSoup` was used to scrape 2500 house listings on [Trulia](https://trulia.com) across five cities in Missouri (Columbia, Kansas City, St. Louis, Springfield, and Lee's Summit).  Data on each listing include things such as crime rate, schools in the area, listing history, tax history, etc.

- [Cleaning](trulia/trulia_cleaning.py): The [csv](trulia/df/df_full.csv) that is the result of the cleaning was formatted in such a way that the categorical variables are in a  one-hot-encoded format (except for the region) and numeric variables were left alone.

- [Analysis](trulia/trulia_analysis.py): The explanation of the variables can be found in the file tagged linked to 'analysis' at the beginning of this block.  House price, price per square foot, total square footage, and lot size is just one example of what was compared across the five cities in Missouri.

- [Model](trulia/trulia_model.py): A linear regression algorithm that was coded from scratch was originally used but the results were all `nan` values, so `sklearn` was used instead.  The R2 score from the model was 0.72, however; the MSE was extremely high.  Perhaps a larger selection of the data (or better cleaning of the results) needs to be done to work better with the linear regression models.

### [Slashdot](slashdot) [WIP]

- [Scraping](slashdot/slashdot.py): `BeautifulSoup` is being used to scrape posts on slashdot to eventually perform a sentiment analysis on the posts' titles, and the comments on the posts.
