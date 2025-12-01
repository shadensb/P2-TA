This project is a Streamlit application built using Python. Below is a summary of the main libraries, functions, and special implementations used in the code.

-------------------------------------
1. Libraries Used
-------------------------------------
- streamlit: for building the web interface
- pandas: for data loading and manipulation
- numpy: for numeric operations and log transformations
- plotly.express: for interactive visualizations
- matplotlib.pyplot: for additional plotting
- seaborn: for heatmaps and styling
- re (regular expressions): for extracting list elements from string-formatted data
- warnings: to suppress unnecessary warnings
- scipy.stats.linregress: imported for regression calculations (not used directly)

-------------------------------------
2. Main Functions and Methods Created
-------------------------------------

(1) load_and_prepare_data()
- Loads the CSV file and handles FileNotFoundError
- Filters movies and drops unused columns
- Cleans missing values (scores, votes, titles, popularity, runtime)
- Creates log-transformed columns using numpy.log1p
- Calculates the Weighted Rating (WR) using a custom formula
- Extracts genres and countries using a regex function
- Explodes lists into rows for better filtering
- Returns the prepared dataframe + benchmark values

(2) extract_list(text)
- Helper method using regex to detect and extract list items inside `' '`
- Handles empty or missing list fields

(3) calculate_weighted_rating()
- Custom function applied row-by-row to compute WR
- Uses IMDb score, vote count, vote threshold (m), and mean benchmark (C)

-------------------------------------
3. Unique or Notable Code Usages
-------------------------------------

- @st.cache_data:
  Used to cache the processed dataset so the app runs faster.

- Log scaling (log1p):
  Applied to votes and popularity values to avoid issues with zero and large ranges.

- Double explode technique:
  First exploding genres, then countries, to allow single-genre and single-country analysis.

- Dynamic color-coded Plotly charts:
  Line charts, bar charts, scatter plots with hover data and templates.

- Quadrant lines in scatter plot:
  Added using Plotly's add_hline() and add_vline() based on filtered mean values.

- Custom column selections:
  Prepares a final set of cleaned columns for consistent UI display.

-------------------------------------
4. Notes
-------------------------------------
- App layout uses Streamlit's sidebar, columns, page config, and markdown.
- All charts respond to filters (genre + year range).
- Duplicate movie entries removed only for preview tables, not for analysis.

