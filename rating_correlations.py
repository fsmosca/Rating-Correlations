"""
A streamlit web app to estimate rating as target like Lichess or Chess.com chess960
based on other rating type like bullet and blitz as features.

Requirements:
    streamlit==1.5.0
    scikit-learn==1.0.2
    xgboost==1.5.2
    statsmodels==0.13.1
    shap==0.40.0
    matplotlib==3.4.3
"""


__version__ = '1.5.1'
__author__ = 'fsmosca'
__script_name__ = 'rating_correlations'
__about__ = 'A streamlit web app to estimate rating as target based on other rating as feature.'


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
from math import sqrt
import statsmodels.api as sm
import shap
import numpy as np
import seaborn as sns


plt.rc("figure", figsize=(12, 6))
plt.rc("font", size=8)


st.set_page_config(
    page_title="Rating Correlations",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'about': f'Rating Correlations v{__version__}'}
)


st.markdown(
"""
<style>
.streamlit-expanderHeader {
    color: SeaGreen
}
.css-qrbaxs {
    color: SlateGray;
    font-weight: normal;
    font-style: italic;
}
.css-1qcggol {
    font-weight: bold;
}
.st-ag {
    font-weight: bold;
}
.css-1cpxqw2 {
    font-weight: bold;
    background-color: #083A30;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


MIN_CHESSCOM_CRAZYHOUSE_RD = 200


@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


@st.cache(allow_output_mutation=True)
def read_file(fn):
    df = pd.read_csv(fn)
    return df


def shap_plot(df, target: str, server):
    """
    Use matplotlib 3.4.3
    target: chess960rating | crazyhouserating
    """
    gt = target.split('rating')[0]  # chess960, crazyhouse ...
    mingames = 50
    minratings = 500
    df = df.loc[
          (df['bulletgames'] >= mingames)
        & (df['bulletrating'] >= minratings)
        & (df['blitzgames'] >= mingames)
        & (df['blitzrating'] >= minratings)
        & (df['rapidgames'] >= mingames)
        & (df['rapidrating'] >= minratings)
    ]

    feature_name = [
        'bulletrating', 'blitzrating', 'rapidrating', 'classicalrating',
        'bulletgames', 'blitzgames', 'rapidgames', 'classicalgames'
    ]

    if server == 'Lichess.org':
        df = df.loc[(df['classicalgames'] >= mingames) & (df['classicalrating'] >= minratings)]
        df = df.loc[(df[f'{gt}games'] >= mingames) & (df[f'{gt}rating'] >= minratings)]
    else:
        feature_name.remove('classicalgames')
        feature_name.remove('classicalrating')

        if gt == 'crazyhouse':
            df = df.loc[(df[f'{gt}rd'] <= MIN_CHESSCOM_CRAZYHOUSE_RD) & (df[f'{gt}rating'] >= minratings)]
        else:
            df = df.loc[(df[f'{gt}games'] >= mingames) & (df[f'{gt}rating'] >= minratings)]

    features = df[feature_name]
    target = df[target]
    X = features
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    model = xg.XGBRegressor(objective ='reg:squarederror', booster='gbtree', n_estimators = 2000, seed = 123, n_jobs=1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.markdown(f'''
    #### The impact of features on {gt.title()} Rating prediction
    All datasets used in the regression have {mingames} or more games and {minratings} or more ratings points.  
    Library              : **XGBoost using gbtree**  
    Train Datasets Count : **{len(X_train)}**  
    Test Datasets Count  : **{len(X_test)}**  
    RMSE                 : **{round(rmse)}**  
    ''')

    fig, _ = plt.subplots()
    shap_values_test = shap.TreeExplainer(model).shap_values(X_test)
    shap.summary_plot(shap_values_test, X_test, show=False, plot_size=(8, 4))
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=80)
    st.image(buf)


def show_title(server):
    st.markdown(f'''
    #### {server} Rating Correlations
    ''')
    if server == 'Lichess.org':
        return 'lichess_chess960_crazyhouse.csv'
    return 'chesscom_chess960_crazyhouse.csv'


def plot_2dhist(df, server, game_type, feature):
    dfx = df.loc[(abs(df[f'{feature}rating'] - df[f'{game_type}rating']) <= 400)]
    plt.figure(figsize=(6,4))
    sns.histplot(
        dfx, x=f'{feature}rating', y=f'{game_type}rating',
        bins=20, discrete=(False, False), log_scale=(False, False),
        cbar=True, cbar_kws=dict(shrink=0.8),
    )
    plt.tight_layout()
    plt.xlabel(f'{server} {feature} rating', fontsize=9)
    plt.ylabel(f'{server} {game_type} rating', fontsize=9)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    st.image(buf)
    plt.close()   


def dist_plot(server, game_type):
    """
    server: lichess.org or chess.com
    game_type: chess960 or crazyhouse
    """
    fn = None
    if server == 'Lichess.org':
        fn = 'lichess_chess960_crazyhouse.csv'
    elif server == 'Chess.com':
        fn = 'chesscom_chess960_crazyhouse.csv'
    else:
        raise Exception(f'server {server} is not defined.')

    df = read_file(fn)
    if game_type == 'chess960':
        df = df.loc[(df[f'{game_type}games'] >= 50) & (df[f'{game_type}rating'] >= 500)]
    elif game_type == 'crazyhouse':
        if server == 'Lichess.org':
            df = df.loc[(df[f'{game_type}games'] >= 50) & (df[f'{game_type}rating'] >= 500)]
        elif server == 'Chess.com':
            df = df.loc[(df[f'{game_type}rd'] <= 200) & (df[f'{game_type}rating'] >= 500)]

    stats = {
        'mean': round(df[f"{game_type}rating"].mean()),
        'mode': round(df[f"{game_type}rating"].mode().iat[0]),
        'median': round(df[f"{game_type}rating"].median()),
        'stdev': round(df[f"{game_type}rating"].std()),
        'datasets': df.shape[0]
    }

    st.markdown(f'''
    ##### {server} - {game_type}
    ''')
    st.write(stats)
    plt.figure(figsize=(8,4))
    bin = 50
    sns.displot(df[f'{game_type}rating'], bins=bin, kde=True)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    st.image(buf)
    plt.close()

    for f in ['bullet', 'blitz', 'rapid']:
        plot_2dhist(df, server, game_type, f)


def build_model(reg_type, multi_features, X_train, y_train, X_test, y_test):
    if reg_type == 'xgboost':
        model = xg.XGBRegressor(objective ='reg:squarederror', booster='gblinear', n_estimators = 2000, seed = 123, n_jobs=1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        y_pred = model.predict(X_test)
        coeff = {'const': model.intercept_[0]}
        for i, f in enumerate(multi_features):
            coeff.update({f: model.coef_[i]})
    elif reg_type == 'statsmodels':
        X_train = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit()
        dfres = model.params
        dfres = dfres.reset_index()
        coeff = dict(dfres.values)  # keys = const, feature1, ...
        X_test_tmp = X_test.copy()
        X_test_tmp = sm.add_constant(X_test_tmp)
        y_pred = model.predict(X_test_tmp)
    return model, y_pred, coeff


def main():
    # Initial session states
    if 'bulletrating' not in st.session_state:
        st.session_state.bulletrating = 1000

    if 'blitzrating' not in st.session_state:
        st.session_state.blitzrating = 1000

    if 'rapidrating' not in st.session_state:
        st.session_state.rapidrating = 1000

    if 'classicalrating' not in st.session_state:
        st.session_state.classicalrating = 1000

    if 'target_type' not in st.session_state:
        st.session_state.target_type = 'Chess960'
    
    if 'regmingames' not in st.session_state:
        st.session_state.regmingames = 50

    if 'regminrating' not in st.session_state:
        st.session_state.regminrating = 1000

    if 'regoutlier' not in st.session_state:
        st.session_state.regoutlier = 400    

    if 'plot_distribution' not in st.session_state:
        st.session_state.plot_distribution = False

    if 'plot_shap' not in st.session_state:
        st.session_state.plot_shap = False

    st.sidebar.write('# REGRESSION OPTIONS')

    server = st.sidebar.selectbox('Select Server', ['Lichess.org', 'Chess.com'])
    fn = show_title(server)
    df = read_file(fn)
    if server == 'Chess.com':
        df.columns = ['username', 'chess960games', 'chess960rating',
                      'bulletgames', 'bulletrating', 'blitzgames', 'blitzrating',
                      'rapidgames', 'rapidrating', 'crazyhouserd', 'crazyhouserating'
        ]

    df1 = df.copy()
    df2 = df.copy()

    reg_type = st.sidebar.selectbox('Select Regressor', ['statsmodels', 'xgboost'])

    multi_feature_options = ['Bullet', 'Blitz', 'Rapid', 'Classical']
    if server == 'Chess.com':
        multi_feature_options.remove('Classical')

    multi_features = st.sidebar.multiselect(
        label='Select Features', 
        default=['Bullet'],
        options=multi_feature_options
    )

    target_options = ['Chess960', 'Bullet', 'Blitz', 'Rapid', 'Classical', 'Crazyhouse']
    if server == 'Chess.com':
        target_options.remove('Classical')

    for feat in multi_features:
        if feat in target_options:
            target_options.remove(feat)

    multi_features = [f'{m.lower()}rating' for m in multi_features]

    st.sidebar.selectbox(
        label='Select Target',
        options=target_options,
        key='target_type'
    )
    target_type_lower = st.session_state.target_type.lower()

    st.sidebar.number_input(
        label='Min Games',
        min_value=50,
        max_value=1000,
        key='regmingames',
        help='default=50, min=50, max=1000, if server is Chess.com '
             ' and target is crazyhouse, the RD must be 150 or less'
    )

    st.sidebar.number_input(
        label='Min Rating',
        min_value=1000,
        max_value=2200,
        key='regminrating',
        help='default=1000, min=1000, max=2200'
    )

    st.sidebar.number_input(
        label='Outlier Margin',
        min_value=50,
        max_value=1000,
        key='regoutlier',
        help='default=400, min=50, max=1000, the minimum difference between feature rating and target rating, '
             'if low the error will be lower, but would result to lesser data points usage'
    )

    st.sidebar.checkbox(label='Plot Distributions', key='plot_distribution')
    st.sidebar.checkbox(label='Plot SHAP', key='plot_shap')

    if len(multi_features) == 0:
        st.warning('Please select a feature.')
        st.stop()

    # Remove data that does not meet the minimum games and ratings.
    for f in multi_features:
        gt = f.split('rating')[0]  # bullet, blitz ...
        df1 = df1.loc[df1[f'{gt}games'] >= st.session_state.regmingames]
        df1 = df1.loc[df1[f'{gt}rating'] >= st.session_state.regminrating]

    if server == 'Chess.com' and target_type_lower == 'crazyhouse':
        df1 = df1.loc[df1[f'{target_type_lower}rd'] <= MIN_CHESSCOM_CRAZYHOUSE_RD]
    else:
        df1 = df1.loc[df1[f'{target_type_lower}games'] >= st.session_state.regmingames]
    df1 = df1.loc[df1[f'{target_type_lower}rating'] >= st.session_state.regminrating]

    # Remove outliers by rating diff.
    df1 = df1.loc[
          (abs(df1[f'{target_type_lower}rating'] - df1['bulletrating']) <= st.session_state.regoutlier)
        & (abs(df1[f'{target_type_lower}rating'] - df1['blitzrating']) <= st.session_state.regoutlier)
        & (abs(df1[f'{target_type_lower}rating'] - df1['rapidrating']) <= st.session_state.regoutlier)
    ]
    if server == 'Lichess.org':
        df1 = df1.loc[(abs(df1[f'{target_type_lower}rating'] - df1['classicalrating']) <= st.session_state.regoutlier)]

    df1 = df1.reset_index(drop=True)

    features = df1[multi_features]
    target = df1[f'{target_type_lower}rating']

    X = features
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    model, y_pred, coeff = build_model(reg_type, multi_features, X_train, y_train, X_test, y_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    with st.expander('CONVERSION', expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with st.form(key='conversion_form_k'):
            with col1:
                # Bullet
                maxv = df1['bulletrating'].max()
                bullet_disabled = False if 'bulletrating' in multi_features else True
                st.number_input(
                    label=f'Input Bullet Rating',
                    min_value=1000,
                    max_value=maxv,
                    key='bulletrating',
                    disabled=bullet_disabled,
                    help=f'min=1000, max={maxv}'
                )
            with col2:
                # Blitz
                maxv = df1['blitzrating'].max()
                blitz_disabled = False if 'blitzrating' in multi_features else True
                st.number_input(
                    label=f'Input Blitz Rating',
                    min_value=1000,
                    max_value=maxv,
                    key='blitzrating',
                    disabled=blitz_disabled,
                    help=f'min=1000, max={maxv}'
                )
            with col3:
                # Rapid
                maxv = df1['rapidrating'].max()
                rapid_disabled = False if 'rapidrating' in multi_features else True
                st.number_input(
                    label=f'Input Rapid Rating',
                    min_value=1000,
                    max_value=maxv,
                    key='rapidrating',
                    disabled=rapid_disabled,
                    help=f'min=1000, max={maxv}'
                )
            with col4:
                # Classical
                if server != 'Chess.com':
                    maxv = df1['classicalrating'].max()
                classical_disabled = False if 'classicalrating' in multi_features and server != 'Chess.com' else True
                st.number_input(
                    label=f'Input Classical Rating',
                    min_value=1000,
                    max_value=maxv,
                    key='classicalrating',
                    disabled=classical_disabled,
                    help=f'min=1000, max={maxv}'
                )  

            if reg_type == 'statsmodels':
                target_rating = coeff['const']
                if 'bulletrating' in multi_features:
                    target_rating += st.session_state.bulletrating * coeff['bulletrating']
                if 'blitzrating' in multi_features:
                    target_rating += st.session_state.blitzrating * coeff['blitzrating']
                if 'rapidrating' in multi_features:
                    target_rating += st.session_state.rapidrating * coeff['rapidrating']
                if 'classicalrating' in multi_features:
                    target_rating += st.session_state.classicalrating * coeff['classicalrating']
            elif reg_type == 'xgboost':
                target_rating = coeff['const']
                for f in multi_features:
                    target_rating += st.session_state[f] * coeff[f]

            pred_interval = round(1.96*rmse)
            is_calculate_rating = st.form_submit_button(label='Calculate Rating Prediction')
            prediction_value = 'None'
            if is_calculate_rating:
                prediction_value = f'{round(target_rating)}'
            st.markdown(f'''
            Rating Prediction: **{prediction_value}**, Type: **{st.session_state.target_type}**, Margin of Error: **+/- ({pred_interval})**, Confidence Level: **{95}%**
            ''')

    if not is_calculate_rating:
        return

    # Plot outputs
    with st.expander('REGRESSION PLOT'):
        if reg_type == 'statsmodels':
            isbullet = False
            if 'bulletrating' in multi_features:
                isbullet = True
                fig = sm.graphics.plot_regress_exog(model, "bulletrating")
                fig.tight_layout(pad=1.0)
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

            isblitz = False
            if 'blitzrating' in multi_features:
                isblitz = True
                if isbullet:
                    st.markdown(''' --- ''')
                fig = sm.graphics.plot_regress_exog(model, "blitzrating")
                fig.tight_layout(pad=1.0)
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

            if 'rapidrating' in multi_features:
                if isblitz:
                    st.markdown(''' --- ''')
                fig = sm.graphics.plot_regress_exog(model, "rapidrating")
                fig.tight_layout(pad=1.0)
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

    with st.expander("REGRESSION RESULT"):
        if reg_type == 'statsmodels':
            st.write(model.summary())

    if st.session_state.plot_distribution:
        with st.expander('DISTRIBUTION PLOTS'):
            st.write('Each user has a minimum of 50 games and a minimum rating of 500. When server '
                      'is chess.com and variant is crazyhouse the RD (rating deviation) is 200 and below.')
            col1, col2 = st.columns(2)
            with col1:
                dist_plot('Lichess.org', 'chess960')
            with col2:
                dist_plot('Chess.com', 'chess960')
            st.markdown('''
            ---
            ''')
            col1, col2 = st.columns(2)
            with col1:
                dist_plot('Lichess.org', 'crazyhouse')
            with col2:
                dist_plot('Chess.com', 'crazyhouse')

    if st.session_state.plot_shap:
        with st.expander('SHAP PLOT'):
            tt = 'chess960rating'
            if target_type_lower != 'chess960' and target_type_lower != 'crazyhouse':
                tt = 'chess960rating'
            else:
                tt = f'{target_type_lower}rating'
            shap_plot(df2, tt, server)

    with st.expander("DATASETS"):
        col1, col2 = st.columns(2)

        with col1:
            st.write('Training datasets summary for the given regression variables')            
            st.write(X_train.describe())            
            
        with col2:
            st.write('Test Datasets Summary')
            st.write(X_test.describe())

        st.write('All Datasets Summary')
        st.write(df.describe())

        st.write('Test Datasets Details')
        st.write(df1)

    with st.expander('DOWNLOAD'):
        csv = convert_df(df)
        st.download_button(
            label=f"Download all {server} data as CSV",
            data=csv,
            file_name='rating_correlations.csv',
            mime='text/csv',
        )

    with st.expander('DATA INFO'):
        st.markdown(f"""
        *Date collected:* **2022-01-23 - 2022-02-03**  
        *Source:* **Lichess and Chess.com**  
        **User must have 50 or more games of either chess960 or crazyhouse**  
        **User must have 50 or more games in either bullet, blitz, rapid or classical**   
        **User is not a Lichess Terms of Service violator**
        """
        )

    with st.expander('CREDITS'):
        st.markdown(f"""
        1. BCLC from [stackexchange](https://chess.stackexchange.com)  
        2. [Lichess.org](https://lichess.org/)  
        3. [Chess.com](https://chess.com/)  
        4. [Streamlit](https://github.com/streamlit/streamlit)  
        """
        )        


if __name__ == '__main__':
    main()
