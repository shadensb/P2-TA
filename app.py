import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# Set Matplotlib backend for compatibility
plt.rcParams['figure.max_open_warning'] = 100

# 1.DATA CLEANING AND PREPARATION FUNCTIONS

@st.cache_data
def load_and_prepare_data(file_path='shadensb/p2-ta/main/app.py'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame(), 0, 0 

    # Filter Movies
    movies_df = df[df['type'] == 'MOVIE'].copy()
    
    # Drop columns not in use -  final 11 features:
    cols_to_drop = ['description', 'age_certification', 'seasons', 'tmdb_score', 'imdb_id', 'type']
    #loop it for user pref in each time they filter years or gernres in UI
    movies_df.drop(columns=[c for c in cols_to_drop if c in movies_df.columns], errors='ignore', inplace=True)
    
    # 2. MISSING VALUES
    movies_df.dropna(subset=['imdb_score'], inplace=True) # Drop missing scores
    movies_df['imdb_votes'].fillna(0, inplace=True)     # Fill missing votes with 0
    movies_df['title'].fillna('Title Missing', inplace=True)     # Replace missing title
    movies_df['tmdb_popularity'].fillna(0, inplace=True)     # Replace missing tmdb_popularity with 0
    movies_df.dropna(subset=['runtime'], inplace=True) 

    # Store Original Mean (Benchmark)
    C_benchmark = movies_df['imdb_score'].mean() 
    
    # Create Log Columns for visualization scale
    # We use log1p because imdb_votes and tmdb_popularity now contain 0
    movies_df['log_imdb_votes'] = np.log1p(movies_df['imdb_votes'])
    movies_df['log_tmdb_popularity'] = np.log1p(movies_df['tmdb_popularity'])

    # Calculate Weighted Score (WR) 
    m_threshold = movies_df['imdb_votes'].quantile(0.75) 

    def calculate_weighted_rating(df_row, m, C):
        v = df_row['imdb_votes']
        R = df_row['imdb_score']
        return (v / (v + m) * R) + (m / (v + m) * C)

    movies_df['weighted_score'] = movies_df.apply(calculate_weighted_rating, axis=1, m=m_threshold, C=C_benchmark)

    # Double Explode (Genres & Countries) and Finaliz

    # Regex for extracting list elements
    def extract_list(text):
        if pd.isna(text) or str(text).strip() in ['[]', '']:
            return []
        # Use regex to find all strings inside single quotes
        return re.findall(r"'(.*?)'", str(text))

    # 1. Explode Genres
    movies_df['genre_list'] = movies_df['genres'].apply(extract_list)
    df_working_genres = movies_df.explode('genre_list').copy() #works with actual list not a string list
    df_working_genres['single_genre'] = df_working_genres['genre_list'].str.strip()
    df_working_genres = df_working_genres[df_working_genres['single_genre'].str.len() > 0]
    
    # 2. Explode Countries
    df_working_genres['country_list'] = df_working_genres['production_countries'].apply(extract_list) 
    df_final_exploded = df_working_genres.explode('country_list').copy()
    df_final_exploded['single_country'] = df_final_exploded['country_list'].str.strip()
    df_final_exploded = df_final_exploded[df_final_exploded['single_country'].str.len() > 0]

    # Select only the clean and necessary columns FINAL
    columns_to_keep_final = ['id', 'title', 'release_year', 'runtime', 'imdb_score', 'imdb_votes', 
        'tmdb_popularity', 'genres', 'production_countries', 
        # New Metrics
        'weighted_score', 'log_imdb_votes', 'log_tmdb_popularity', 'single_genre', 'single_country']

    movies_df_final = df_final_exploded[columns_to_keep_final].reset_index(drop=True)

    return movies_df_final, C_benchmark, m_threshold

# Load Data and Benchmarks from prev method
movies_df_final, C_benchmark, m_threshold = load_and_prepare_data()

# Check if data loading was successful
if movies_df_final.empty:
    st.error(
        "🚨 ERROR: The application could not process data. Please ensure 'titles.csv' is in the application directory."
    )
    st.stop()

# STREAMLIT APP UI
st.set_page_config( layout="wide",  page_title="Movie Analysis Dashboard",  page_icon="🎬")

# SIDEBAR: Filters and Description 
st.sidebar.header("Dashboard Description")
st.sidebar.markdown(
    """
    This dashboard analyzes movie production trends, quality trends, and popularity 
    (TMDB Popularity) using the **Weighted Score** to provide an unbiased quality metric.
    
    * **Quality:** Measured by Weighted Score (IMDb Score adjusted for vote count).

    * **Benchmark (C):** Average movie ratings of **{:.2f}** based on IMDb Score.

    * **Vote Threshold (m):** **{:.0f}** votes (75th percentile).
    """.format(C_benchmark, m_threshold))

st.sidebar.header("Filters")

# Filter 1: Release Year Slider
min_year = int(movies_df_final['release_year'].min())
max_year = int(movies_df_final['release_year'].max())
year_range = st.sidebar.slider('Select Release Year Range', min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Filter 2: Genre Multi-select
all_genres = sorted(movies_df_final['single_genre'].unique())
selected_genres = st.sidebar.multiselect('Filter by Genre', all_genres,
    default=all_genres)

# Apply Filters - this df will be used in all plots in UI based on their selection
df_filtered = movies_df_final[
    (movies_df_final['release_year'] >= year_range[0]) & 
    (movies_df_final['release_year'] <= year_range[1]) &
    (movies_df_final['single_genre'].isin(selected_genres))
]


# MAIN PAGE
st.title("🎬 Movie Industry Analysis: Quality, Hype, and Trends")
st.markdown("---")
# 3. Data Preview and Summary Statistics
st.header("1. Data Preview and Summary")

st.space("small") # Adds a small-sized vertical space
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Preview (First 4 Movies)")
    # Create a unique view for display only cause we need the doublicates
    df_unique_view = df_filtered.drop_duplicates(subset=['id'], keep='first')
    
    # Showing the final clean columns
    display_cols = ['title', 'release_year', 'runtime', 'imdb_score', 'weighted_score', 'tmdb_popularity', 'single_genre', 'single_country']
    st.dataframe(df_unique_view[display_cols].head(4))

with col2:
    st.subheader("Summary Statistics")
    stats_cols = ['imdb_score', 'weighted_score', 'tmdb_popularity', 'runtime']
    # Calculate summary statistics on the unique view (df_unique_view) for accurate overall stats
    st.dataframe(df_unique_view[stats_cols].describe().T.apply(lambda x: x.round(2)))
    #T : to flip our table, transpose
st.markdown("---")
st.space("small") 

# Interactive Visualizations
st.header("2. Production Trends and Rankings")
st.space("small")
# Q8: Production Over Time 
st.subheader("Production Volume Over Time (Trend Analysis)")
st.markdown("""Has production increased over time? Is the demand higher than before? """)
production_volume = movies_df_final.groupby('release_year')['id'].nunique().reset_index()
production_volume.columns = ['Year', 'Movie_Count']

# Plotly Line Chart
fig_volume = px.line( production_volume, x='Year', y='Movie_Count', 
title='Movie Production Count by Release Year', height=400, template='plotly_white')
st.plotly_chart(fig_volume, use_container_width=True)
st.markdown("""We observe a noticeable drop in movie production in 2022. However, this decline is most likely due to limitations within the dataset.""")
st.space("small")
st.markdown("---")

# Q1: Top 10 Genres (Movies) and Countries (Movies) 
st.subheader("Top 10 Genres and Countries (Number of Movies)")
st.markdown(""" We must now ask ourselves, which genres and nations are truly dominating the movie market?""")
col_a, col_b = st.columns(2)

with col_a:
    # Top 10 Genres by movie num
    genre_counts = df_filtered['single_genre'].value_counts().head(10).reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    fig_genre_vol = px.bar(genre_counts, x='Genre', y='Count', title='Top 10 Genres by Movie Count', 
    color='Count', color_continuous_scale=px.colors.sequential.Plasma, template='plotly_white')

    st.plotly_chart(fig_genre_vol, use_container_width=True)

    # Insight for Genre num
    st.markdown("""
    **Insight:** We observed that **Drama** and **Comedy** genres consistently have the highest volume of movie production. 
    This trend suggests the market prioritizes easily produced and broadly appealing content.""")

with col_b:
    # Top 10 Countries by movie nm
    country_counts = df_filtered['single_country'].value_counts().head(10).reset_index() # reset changes index(contries to column) for mapping country code
    country_counts.columns = ['Country_Code', 'Count']

    # Map Country Codes for Better Display
    country_map = {'US': 'United States', 'IN': 'India', 'GB': 'UK', 'CA': 'Canada', 'FR': 'France', 'DE': 'Germany', 'JP': 'Japan', 'KR': 'S. Korea', 'MX': 'Mexico', 'ES': 'Spain', 'PH': 'Philippines', 'CN': 'China'}
    country_counts['Country'] = country_counts['Country_Code'].map(lambda x: country_map.get(x, x))
    
    fig_country_vol = px.bar( country_counts, x='Country', y='Count', title='Top 10 Countries by Production Volume', 
    color='Count', color_continuous_scale=px.colors.sequential.Viridis, template='plotly_white')

    st.plotly_chart(fig_country_vol, use_container_width=True)

    # Insight for Country numbr
    st.markdown("""**Insight:** Production volume is heavily concentrated in the **United States** (US), with **India** (IN) being a strong second. This confirms the global cinematic dominance of Hollywood and the massive scale of the Bollywood industry.""")
st.markdown("---") 


#Q3: Country Quality Ranking 
st.subheader("Top 10 Countries by Highest Average Weighted Score")
st.markdown("""Does producing more movies mean better quality, or are smaller movie industries better at making consistently good films?""")

# 1. Calculate Avg Weighted Score per Country (for ranking quality)
country_quality = df_filtered.groupby('single_country')['weighted_score'].mean().sort_values(ascending=False).head(10).reset_index()
country_quality.columns = ['Country_Code', 'Avg_Weighted_Score']

# 2. Map Country Codes to Full Names for Plotly (using the same map defined above)
country_quality['Country'] = country_quality['Country_Code'].map(lambda x: country_map.get(x, x))
country_quality = country_quality.sort_values(by='Avg_Weighted_Score', ascending=False)

# Plot the average Weighted Score
fig_quality_rank = px.bar(country_quality, x='Country', y='Avg_Weighted_Score', title='Top 10 Countries Ranked by Average Weighted Score (Quality)',
    color='Avg_Weighted_Score', color_continuous_scale=px.colors.sequential.Sunset, template='plotly_white')
st.plotly_chart(fig_quality_rank, use_container_width=True)

#Insight for Country Quality
st.markdown("""**Insight:** As shown, we may notice a takeaway that high production volume (Quantity) might have an **inverse correlation** with the average quality of production (Quality). Smaller countries often show better scoring consistency after factoring in vote threshold.""")
st.markdown("---") 


# Q5: Genre Quality vs. Popularity Tables
st.subheader("Top 10 Genres: Quality vs. Popularity Rankings")
col_s, col_p = st.columns(2)

# Q5 Analysis: Aggregate Data for Ranking
genre_averages = df_filtered.groupby('single_genre').agg(Avg_IMDb_Score=('imdb_score', 'mean'),
Avg_TMDB_Popularity=('tmdb_popularity', 'mean'),Total_Count=('id', 'count')).reset_index()

# Rank and Select Top 10 by IMDb Score (Quality)
top_10_score = genre_averages.sort_values(by='Avg_IMDb_Score', ascending=False).head(10).copy()

# Rank and Select Top 10 by TMDB Popularity (Preference/Hype)
top_10_popularity = genre_averages.sort_values(by='Avg_TMDB_Popularity', ascending=False).head(10).copy()


#Display 1: Top 10 by IMDb Score 
with col_s:
    st.markdown("##### 🌟 Top 10 Genres Ranked by Highest Average IMDb Score")
    score_table = top_10_score[['single_genre', 'Avg_IMDb_Score', 'Total_Count', 'Avg_TMDB_Popularity']].rename(columns={
        'single_genre': 'Genre', 
        'Avg_IMDb_Score': 'Avg IMDb Score',
        'Total_Count': 'Movie Count', 
        'Avg_TMDB_Popularity': 'Avg TMDB Popularity'})
    st.table(score_table.round(2)) # Use st.table for small display

#Display 2: Top 10 by TMDB Popularity
with col_p:
    st.markdown("##### 📈 Top 10 Genres Ranked by Highest Average TMDB Popularity")
    popularity_table = top_10_popularity[['single_genre', 'Avg_IMDb_Score', 'Total_Count', 'Avg_TMDB_Popularity']].rename(columns={
        'single_genre': 'Genre', 
        'Avg_IMDb_Score': 'Avg IMDb Score', 
        'Total_Count': 'Movie Count', 
        'Avg_TMDB_Popularity': 'Avg TMDB Popularity'})
    st.table(popularity_table.round(2)) # Use st.table for small display

#  Q5 Insight 
st.markdown("""**Insight:** The comparison between the two rankings shows that highly rated genres are not necessarily the most popular ones.""")
st.markdown("---") 
st.space("small")

# 5. Quality and Correlation Analysis 
st.header("3. Quality and Performance Analysis")
st.space("small")

#Q4: Runtime vs. Quality (Scatter) 
st.subheader("Runtime vs. Quality (Correlation)")
st.markdown("""What is the correlation between Runtime and Ratings per Genre? Does the length of a movie impact how highly its genre is rated, or is there no clear connection?""")

# Get top 5 genres for color-coding the scatter plot
top_5_genres = movies_df_final['single_genre'].value_counts().head(5).index.tolist()
df_scatter = df_filtered[df_filtered['single_genre'].isin(top_5_genres)].copy()

fig_runtime = px.scatter(df_scatter, x='runtime', y='weighted_score', color='single_genre',
    hover_data=['title', 'imdb_score', 'weighted_score', 'runtime'],
    title='Movie Runtime vs. Weighted Score (Top 5 Genres)', labels={'runtime': 'Runtime (Minutes)', 'weighted_score': 'Weighted Score (Quality)'},
    template='plotly_white')
fig_runtime.update_layout(legend_title_text='Genre')
st.plotly_chart(fig_runtime, use_container_width=True)

#  Insight for Runtime Correlation 
st.markdown("""**Insight:** We see a dense cluster of movies across all genres.""")
st.space("small")

# Q6: Overrated/Underrated Scatter Plot 
st.subheader("Hype vs. Quality: Overrated vs. Underrated")
st.space("small")

st.markdown("""Is there a pattern where a certain genras has overrate/underrated movies? This is measured by comparing the correlation between hype and quality.""")
# Calculate the mean lines for the Overrated/Underrated Plot
mean_wr_filter = df_filtered['weighted_score'].mean()
mean_log_pop_filter = df_filtered['log_tmdb_popularity'].mean()

# Filter data to top 5 genres for visualization clarity (same as Q4)
df_hype = df_filtered[df_filtered['single_genre'].isin(top_5_genres)].copy()

fig_hype = px.scatter(df_hype, x='weighted_score', y='log_tmdb_popularity', color='single_genre',
    hover_data=['title', 'weighted_score', 'tmdb_popularity'],
    title='Movie Quality vs. Hype (Weighted Score vs. Log Popularity)',
    labels={'weighted_score': 'Weighted Score (Quality)', 'log_tmdb_popularity': 'Log TMDB Popularity (Hype)'},
    template='plotly_white')

# Add quadrant lines using the filtered data's mean
fig_hype.add_hline(y=mean_log_pop_filter, line_dash="dash", line_color="red",  annotation_text=f"Avg Log Hype ({mean_log_pop_filter:.2f})")
fig_hype.add_vline(x=mean_wr_filter, line_dash="dash", line_color="green",  annotation_text=f"Avg WR ({mean_wr_filter:.2f})")

fig_hype.update_layout(legend_title_text='Genre')
st.plotly_chart(fig_hype, use_container_width=True)

st.markdown("""**Insight:** We notice a great amount of underrated movies in the most watched genras, but greater in overrated movies than those how have fair hype and quality rates. This might be a cause of that a lot of people tend to follow the crowd?""")
st.markdown("---") 


# Q7: Correlation Heatmap
st.subheader("Correlation Heatmap of Key Movie Metrics")
st.space("small")
st.markdown("""Are movie attributes, like length, truly related to the overall quality and hype?""")
st.space("small")

# 1. Prepare data for correlation
correlation_data = df_filtered[['imdb_score', 'weighted_score', 'tmdb_popularity', 'imdb_votes','runtime']].copy()
correlation_data.columns = ['IMDb Score (R)', 'Weighted Score (WR)', 'TMDB Popularity', 'IMDb Votes','Runtime (min)']

# 2. Calculate Correlation Matrix
correlation_matrix = correlation_data.corr()

# 3. Start Plotting the Heatmap using Matplotliba nd Seaborn
fig_corr, ax_corr = plt.subplots(figsize=(8, 7)) # Create figure and axis

sns.heatmap(correlation_matrix, 
    annot=True,          # Display the correlation values
    cmap='coolwarm',     # Color map for visualizing pos/neg correlations
    fmt=".2f",           # Format to two decimal places
    linewidths=.5,       # Lines between cells
    cbar_kws={'label': 'Correlation Coefficient'},
    ax=ax_corr           # Pass the Matplotlib axis
    )

ax_corr.set_title('Correlation Heatmap of Key Movie Metrics', fontsize=14)
ax_corr.tick_params(axis='x', rotation=45)
ax_corr.tick_params(axis='y', rotation=0)
plt.tight_layout()

st.pyplot(fig_corr) # Display the Matplotlib figure

#  Q7 Insight 
st.markdown("""**Insight:** The heatmap confirms a near-perfect correlation between **IMDb Score** and **Weighted Score** (WR), validating our quality metric. Critically, **Runtime** shows only a marginal positive correlation with quality, indicating that movie length is not a major factor in determining a film's score.""")

st.markdown("---") 


# MoreTables
st.header("4. Other Rankings")
st.space("small")

col_u, col_o = st.columns(2)

#  UNDERRATED MOVIES (High Quality / Low Hype)\
with col_u:
    st.subheader("💎 Top 5 Underrated Movies")
    
    # Calculate the Underrated Rank (WR - Log Pop difference)
    underrated_df = df_filtered[ (df_filtered['weighted_score'] > mean_wr_filter) & (df_filtered['log_tmdb_popularity'] < mean_log_pop_filter)
    ].copy()
    
    # Drop duplicates to count each unique movie once
    if not underrated_df.empty:
        underrated_df = underrated_df.sort_values(by='imdb_votes', ascending=False).drop_duplicates(subset=['id'], keep='first').copy()
        
        underrated_df['underrated_rank'] = underrated_df['weighted_score'] - underrated_df['log_tmdb_popularity']
        top_underrated = underrated_df.sort_values(by='underrated_rank', ascending=False).head(5)
        
        underrated_table = top_underrated[[ 'title', 'single_genre', 'weighted_score', 'tmdb_popularity'
        ]].rename(columns={'title': 'Title', 'single_genre': 'Genre','weighted_score': 'WR','tmdb_popularity': 'Popularity'
        }).reset_index(drop=True)
        st.table(underrated_table.round(2))
    else:
        st.info("No underrated movies found for the current filter selection.")


#  OVERRATED MOVIES (Low Quality / High Hype)
with col_o:
    st.subheader("💣 Top 5 Overrated Movies")
    
    # Calculate the Overrated Rank (Log Pop - WR difference)
    overrated_df = df_filtered[(df_filtered['weighted_score'] < mean_wr_filter) & (df_filtered['log_tmdb_popularity'] > mean_log_pop_filter)
    ].copy()
    
    if not overrated_df.empty:
        # Drop duplicates to count each unique movie once
        overrated_df = overrated_df.sort_values(by='imdb_votes', ascending=False).drop_duplicates(subset=['id'], keep='first').copy()

        overrated_df['overrated_rank'] = overrated_df['log_tmdb_popularity'] - overrated_df['weighted_score']
        top_overrated = overrated_df.sort_values(by='overrated_rank', ascending=False).head(5)

        overrated_table = top_overrated[['title', 'single_genre', 'weighted_score', 'tmdb_popularity'
        ]].rename(columns={'title': 'Title', 'single_genre': 'Genre','weighted_score': 'WR','tmdb_popularity': 'Popularity'
        }).reset_index(drop=True)
        st.table(overrated_table.round(2))
    else:
        st.info("No overrated movies found for the current filter selection.")

st.markdown("---")
