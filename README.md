# Rating Correlations

Generates an estimate of the target or prediction rating based on the given feature ratings. It does not predict ratings across different servers. What it can do is on the same server, for example given Lichess bullet and blitz ratings it will predict a chess960 or crazyhouse rating intended for Lichess. Supported servers are Lichess.org and Chess.com.

It has 2 libraries namely xgboost and statsmodels to process single and multiple regression. This web app is written in Python with [Streamlit](https://streamlit.io/) and is hosted by [Heroku](https://www.heroku.com).

Heroku link: https://ratingcorrelations.herokuapp.com/

Data are generated using the [lichess api](https://lichess.org/api) and both the [official](https://www.chess.com/news/view/published-data-api) and [unofficial](https://www.chess.com/clubs/forum/view/guide-unofficial-api-documentation) chess.com apis.

**Usage** <video src='https://user-images.githubusercontent.com/22366935/152262366-fa92f278-17eb-4ede-93e8-02654c3aeb08.mov'/>


### Credits
* [Lichess.org](https://lichess.org/)  
* [Chess.com](https://www.chess.com/)  
* [Streamlit](https://github.com/streamlit/streamlit)  
* [SHAP](https://github.com/slundberg/shap)
* [Heroku](https://www.heroku.com)
