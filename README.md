# Rating Correlations

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://docs.streamlit.io/) 
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

Generates a linear model and use it to predict a rating like chess960 or crazyhouse given bullet and blitz ratings for example. It can only predict ratings from the same server like if your input is from Lichess its prediction is only applicable for Lichess. Supported servers are Lichess and Chess.com.

The users are provided with two libraries namely xgboost and statsmodels to create simple and multiple linear regression models. Regression plots are generated when the libary used is statsmodels. There is also SHAP plot to see which features matters most and in what directions and how much it influences the prediction output. 

Data are generated using the [lichess api](https://lichess.org/api) and both the [official](https://www.chess.com/news/view/published-data-api) and [unofficial](https://www.chess.com/clubs/forum/view/guide-unofficial-api-documentation) chess.com apis. This web app is written in Python with [Streamlit](https://streamlit.io/) and is hosted by [Heroku](https://www.heroku.com). 

Heroku link: https://ratingcorrelations.herokuapp.com/

You can run this locally by cloning this repo, install the dependents in requirements.txt and execute `streamlit run rating_correlations.py`.

**Usage**  
A sample video on how to use it.
<video src='https://user-images.githubusercontent.com/22366935/152673467-8503b09f-85dc-40e8-8455-cdbac694a773.mov'/>

### Credits
* [Lichess.org](https://lichess.org/)  
* [Chess.com](https://www.chess.com/)  
* [Streamlit](https://github.com/streamlit/streamlit)  
* [SHAP](https://github.com/slundberg/shap)
* [Heroku](https://www.heroku.com)
