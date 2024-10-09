NSE Intraday Stock Market Price Prediction

This project is an NSE intraday stock market price prediction tool built using Python and Streamlit. The application allows users to select a stock from the NSE and receive real-time and historical data predictions. The models used for prediction include linear regression, multiple regression, and random forest.

Features:
Real-time data fetching: Stock data is pulled in real-time using the yfinance library.
Historical data integration: Historical stock data is used for building prediction models.
Multiple Stock Market Indicators: Utilizes various stock market indicators (Moving average, bollinger bands etc) as input data to enhance prediction accuracy.
Hourly Predictions: Predict stock prices at hourly intervals throughout the trading day.
Output Comparison: The output displays predicted price at each time point, along with the error ratio.
User Input: The user can enter a stock symbol via the Streamlit front-end interface.
Multiple Models: Predictions are generated using different machine learning models (linear regression, multiple regression, random forest) for comparison.

Tech Stack:
Python
Streamlit (for frontend and user interface)
yfinance (for data fetching)
Machine learning models (linear regression, multiple regression, random forest)
Visual Studio (development environment)

How to Run:
Clone the repository.
Install the required dependencies 
Run the Streamlit app: streamlit run app.py.
Enter the desired stock symbol and get hourly price predictions for the selected stock.
