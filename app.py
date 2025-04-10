#################################################################################################
# File Name - app.py
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This file contains the code for the Flask application that serves the frontend
#################################################################################################

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import click
import time
import requests


# Initialize Flask app  and set up logging  

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################################################################################
# Class Name - IndianStockAnalyzer
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This class contains the logic for analyzing Indian stocks using technical indicators
#################################################################################################

class IndianStockAnalyzer:
   

    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_free_rate: float = 0.05):
        """
        Initialize the Indian Stock Analyzer.
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.stock_info = None
        self.recommendationTopGainer = []

#################################################################################################
# Function Name - fetch_stock_data
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function fetches stock data from Yahoo Finance
#################################################################################################

    def fetch_stock_data(self, symbol: str, exchange: str = "NS", 
                        start_date: str = None,
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
            stock_symbol = f"{symbol}.{exchange}"
            stock = yf.Ticker(stock_symbol)
            
            data = stock.history(start=start_date, interval=interval)
            
            
            if data.empty:
                raise ValueError(f"No data found for symbol {stock_symbol}")
                
            self.stock_data = data
            self.stock_info = stock.info
            
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise
    # def recomendationTopGainer(self) -> list:
    #     recomendationTopGainers = []
    #     return recomendationTopGainers

################################################################################################
# Function Name - fetch_fundamental_data
# Author - Ojas Ulhas Dighe (Updated)
# Date - 28th Mar 2025
# Description - This function fetches fundamental data for a given stock symbol using web scraping and APIs
#              and performs comprehensive fundamental analysis.
#              It uses yfinance for basic fundamental data and additional metrics.
#              The function returns a dictionary containing the fundamental data and analysis results.
#              The scoring criteria for fundamental metrics are defined within the function.
#################################################################################################


    def fetch_fundamental_data(self, symbol: str) -> dict:
        """
    Fetch fundamental data for a given stock symbol using web scraping and APIs.
    
    Parameters:
    symbol (str): Stock symbol to analyze
    
    Returns:
    dict: Comprehensive fundamental analysis data
    """
        try:
            # Construct full symbol for NSE
            full_symbol = f"{symbol}.NS"
            
            # Use yfinance for basic fundamental data
            stock = yf.Ticker(full_symbol)
            info = stock.info
            website = info.get('website', '')
            
            domain = website.split('//')[-1].split('/')[0]
            logoURL = f"https://logo.clearbit.com/{domain}"
            
            # Fetch additional fundamental details
            fundamental_data = {
                'basic_info': {
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'company_logo':info.get('logo_url', 'N/A'),
                    'logoURL' : logoURL
                },
                'valuation_metrics': {
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                },
                'financial_health': {
                    'total_revenue': info.get('totalRevenue', 0),
                    'gross_profit': info.get('grossProfit', 0),
                    'net_income': info.get('netIncomeToCommon', 0),
                    'total_debt': info.get('totalDebt', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'return_on_equity': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                },
                'growth_metrics': {
                    'revenue_growth': info.get('revenueGrowth', 0) * 100,
                    'earnings_growth': info.get('earningsGrowth', 0) * 100,
                    'profit_margins': info.get('profitMargins', 0) * 100
                }
            }
        
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}

################################################################################################
# Function Name - perform_fundamental_analysis
# Author - Ojas Ulhas Dighe (Updated)
# Date - 28th Mar 2025
# Description - This function performs comprehensive fundamental analysis on a given stock symbol.
#             It uses the fetch_fundamental_data function to get the data and then applies scoring criteria
#            to generate a recommendation.
#            The scoring criteria for fundamental metrics are defined within the function. 
# #################################################################################################

    def perform_fundamental_analysis(self, symbol: str) -> dict:
        """
        Perform comprehensive fundamental analysis.
        
        Parameters:
        symbol (str): Stock symbol to analyze
        
        Returns:
        dict: Fundamental analysis insights and recommendations
        """
        try:
            fundamental_data = self.fetch_fundamental_data(symbol)
            
            # Define fundamental analysis scoring criteria
            def score_fundamental_metrics(data):
                score = 0
                
                # PE Ratio Scoring
                pe_ratio = data['valuation_metrics']['pe_ratio']
                if 0 < pe_ratio < 15:
                    score += 2  # Undervalued
                elif 15 <= pe_ratio <= 25:
                    score += 1  # Fairly valued
                else:
                    score -= 1  # Potentially overvalued
                
                # Debt to Equity Scoring
                debt_to_equity = data['financial_health']['debt_to_equity']
                if debt_to_equity < 0.5:
                    score += 2  # Low debt
                elif debt_to_equity < 1:
                    score += 1  # Moderate debt
                else:
                    score -= 1  # High debt
                
                # Growth Metrics Scoring
                revenue_growth = data['growth_metrics']['revenue_growth']
                earnings_growth = data['growth_metrics']['earnings_growth']
                if revenue_growth > 10 and earnings_growth > 10:
                    score += 2  # Strong growth
                elif revenue_growth > 5 and earnings_growth > 5:
                    score += 1  # Moderate growth
                else:
                    score -= 1  # Low growth
                
                # Profitability Scoring
                profit_margins = data['growth_metrics']['profit_margins']
                if profit_margins > 15:
                    score += 2  # High profitability
                elif profit_margins > 10:
                    score += 1  # Good profitability
                else:
                    score -= 1  # Low profitability
                
                # Dividend Yield Scoring
                dividend_yield = data['valuation_metrics']['dividend_yield']
                if dividend_yield > 3:
                    score += 1  # Good dividend
                
                return score
            
            # Generate fundamental recommendation
            fundamental_score = score_fundamental_metrics(fundamental_data)
            
            if fundamental_score >= 4:
                recommendation = "Strong Buy"
            elif fundamental_score >= 2:
                recommendation = "Buy"
            elif fundamental_score >= 0:
                recommendation = "Hold"
            elif fundamental_score >= -2:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
            
            fundamental_data['recommendation'] = recommendation
            fundamental_data['score'] = fundamental_score
            
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {str(e)}")
            return {}



#################################################################################################
# Function Name - calculate_rsi
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Relative Strength Index (RSI) technical indicator
#################################################################################################

    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

#################################################################################################
# Function Name - calculate_macd
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Moving Average Convergence Divergence (MACD) technical indicator
#################################################################################################

    def calculate_macd(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

#################################################################################################
# Function Name - calculate_bollinger_bands
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Bollinger Bands technical indicator
#################################################################################################

    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return upper_band, middle_band, lower_band

#################################################################################################
# Function Name - calculate_technical_indicators
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates various technical indicators
#################################################################################################

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.stock_data.copy()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Volume Indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        return df
#################################################################################################
# Function Name - calculate_volume_indicators
# Date - 18th Mar 2025
# Description - This function calculates volume-based indicators for price action analysis
#################################################################################################

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators for price action analysis."""
        df = data.copy()
        
        # Volume Moving Average
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume Relative to Average (Volume Ratio)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Get positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Calculate MFI
        positive_mf_14 = positive_flow.rolling(window=14).sum()
        negative_mf_14 = negative_flow.rolling(window=14).sum()
        
        money_ratio = positive_mf_14 / negative_mf_14
        df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        # Accumulation/Distribution Line
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)  # Handle division by zero
        ad = clv * df['Volume']
        df['ADL'] = ad.cumsum()
        
        # Chaikin Money Flow (CMF)
        df['CMF'] = (ad.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum())
        
        # Price Rate of Change (ROC)
        df['Price_ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        return df


#################################################################################################
# Function Name - identify_support_resistance
# Date - 18th Mar 2025
# Description - This function identifies support and resistance levels using various methods
#################################################################################################

    def identify_support_resistance(self, data: pd.DataFrame, window: int = 20, 
                                   num_levels: int = 3) -> tuple[list, list]:
        """
        Identify support and resistance levels using various methods.
        
        Parameters:
        data (pd.DataFrame): Price data with OHLC
        window (int): Window size for local extrema detection
        num_levels (int): Number of levels to return
        
        Returns:
        tuple[list, list]: Support and resistance levels
        """
        df = data.copy()
        
        # Method 1: Local highs and lows
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()
        
        # Find local peaks in highs (resistance)
        resistance_levels = []
        for i in range(window, len(df) - window):
            if highs.iloc[i] == df['High'].iloc[i] and \
               all(highs.iloc[i] >= highs.iloc[i-window:i]) and \
               all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
                resistance_levels.append(df['High'].iloc[i])
        
        # Find local troughs in lows (support)
        support_levels = []
        for i in range(window, len(df) - window):
            if lows.iloc[i] == df['Low'].iloc[i] and \
               all(lows.iloc[i] <= lows.iloc[i-window:i]) and \
               all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
                support_levels.append(df['Low'].iloc[i])
        
        # Method 2: Pivot Points
        # Calculate pivot points for most recent data
        last_day = df.iloc[-1]
        pivot = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
        
        r1 = 2 * pivot - last_day['Low']
        r2 = pivot + (last_day['High'] - last_day['Low'])
        
        s1 = 2 * pivot - last_day['High']
        s2 = pivot - (last_day['High'] - last_day['Low'])
        
        resistance_levels.extend([r1, r2])
        support_levels.extend([s1, s2])
        
        # Method 3: Key psychological levels
        # Find key psychological levels (round numbers)
        current_price = df['Close'].iloc[-1]
        
        # Determine magnitude of price
        magnitude = 10 ** (len(str(int(current_price))) - 1)
        
        # Add psychological levels
        for mult in range(1, 11):
            level = mult * magnitude
            if abs(current_price - level) / current_price < 0.15:  # Within 15% of current price
                if level < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
        
        # Remove duplicates and sort
        support_levels = sorted(list(set([round(x, 2) for x in support_levels])), reverse=True)
        resistance_levels = sorted(list(set([round(x, 2) for x in resistance_levels])))
        
        # Return the closest levels to current price
        current_price = df['Close'].iloc[-1]
        
        # Sort support levels by distance from current price (descending)
        support_levels = sorted(support_levels, key=lambda x: abs(current_price - x))
        
        # Sort resistance levels by distance from current price (ascending)
        resistance_levels = sorted(resistance_levels, key=lambda x: abs(current_price - x))
        
        return support_levels[:num_levels], resistance_levels[:num_levels]

#################################################################################################
# Function Name - analyze_price_action
# Date - 18th Mar 2025
# Description - This function analyzes price action patterns and market context
#################################################################################################

    def analyze_price_action(self, data: pd.DataFrame) -> dict:
        """
        Analyze price action patterns and market context.
        
        Parameters:
        data (pd.DataFrame): Price data with OHLC and volume indicators
        
        Returns:
        dict: Price action analysis results
        """
        df = data.copy()
        
        # Get last 5 days of data for pattern analysis
        recent_data = df.tail(5)
        
        # Calculate price action metrics
        body_sizes = abs(recent_data['Close'] - recent_data['Open'])
        wicks_upper = recent_data['High'] - recent_data[['Open', 'Close']].max(axis=1)
        wicks_lower = recent_data[['Open', 'Close']].min(axis=1) - recent_data['Low']
        
        # Average metrics
        avg_body = body_sizes.mean()
        avg_upper_wick = wicks_upper.mean()
        avg_lower_wick = wicks_lower.mean()
        
        # Calculate recent volatility
        recent_volatility = df['Close'].pct_change().rolling(window=5).std().iloc[-1] * 100
        
        # Identify candlestick patterns
        patterns = []
        
        # Last day pattern
        last_day = df.iloc[-1]
        prev_day = df.iloc[-2]
        
        # Doji
        if abs(last_day['Open'] - last_day['Close']) / (last_day['High'] - last_day['Low']) < 0.1:
            patterns.append("Doji")
        
        # Hammer
        if (last_day['Low'] < last_day[['Open', 'Close']].min()) and \
           (wicks_lower.iloc[-1] > 2 * body_sizes.iloc[-1]) and \
           (wicks_upper.iloc[-1] < 0.5 * body_sizes.iloc[-1]):
            patterns.append("Hammer")
        
        # Shooting Star
        if (last_day['High'] > last_day[['Open', 'Close']].max()) and \
           (wicks_upper.iloc[-1] > 2 * body_sizes.iloc[-1]) and \
           (wicks_lower.iloc[-1] < 0.5 * body_sizes.iloc[-1]):
            patterns.append("Shooting Star")
        
        # Engulfing Patterns
        if (last_day['Open'] > prev_day['Close']) and (last_day['Close'] < prev_day['Open']):
            patterns.append("Bearish Engulfing")
        elif (last_day['Open'] < prev_day['Close']) and (last_day['Close'] > prev_day['Open']):
            patterns.append("Bullish Engulfing")
        
        # Identify trend
        sma20 = df['SMA_20'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]
        
        if df['Close'].iloc[-1] > sma20 > sma50:
            trend = "Strong Uptrend"
        elif df['Close'].iloc[-1] > sma20 and sma20 < sma50:
            trend = "Possible Trend Reversal (Bullish)"
        elif df['Close'].iloc[-1] < sma20 < sma50:
            trend = "Strong Downtrend"
        elif df['Close'].iloc[-1] < sma20 and sma20 > sma50:
            trend = "Possible Trend Reversal (Bearish)"
        else:
            trend = "Sideways/Consolidation"
        
        # Volume analysis
        volume_signal = ""
        if df['Volume'].iloc[-1] > df['Volume_SMA_20'].iloc[-1] * 1.5:
            if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
                volume_signal = "High Volume Bullish Day"
            else:
                volume_signal = "High Volume Bearish Day"
        
        # Momentum
        if df['MFI'].iloc[-1] < 20:
            momentum = "Oversold"
        elif df['MFI'].iloc[-1] > 80:
            momentum = "Overbought"
        else:
            momentum = "Neutral"
        
        # Create result dictionary
        result = {
            'trend': trend,
            'patterns': patterns,
            'momentum': momentum,
            'volume_signal': volume_signal,
            'recent_volatility': round(recent_volatility, 2),
            'avg_body_size': round(avg_body, 2),
            'avg_upper_wick': round(avg_upper_wick, 2),
            'avg_lower_wick': round(avg_lower_wick, 2)
        }
        
        return result

#################################################################################################
# Function Name - calculate_technical_indicators
# Date - 18th Mar 2025
# Description - This function calculates various technical indicators
#################################################################################################

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.stock_data.copy()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Volume Indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        # Add volume and price action indicators
        df = self.calculate_volume_indicators(df)
        
        return df


#################################################################################################
# Function Name - calculate_risk_metrics
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates various risk metrics
#################################################################################################

    def calculate_risk_metrics(self) -> dict:
        """Calculate various risk metrics."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        
        # Sharpe Ratio
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Beta (using Nifty 50 as benchmark) 
        try:
            nifty = yf.download('^NSEI', start=self.stock_data.index[0])
            nifty_returns = nifty['Close'].pct_change().dropna()
            beta = np.cov(daily_returns, nifty_returns)[0][1] / np.var(nifty_returns)
        except:
            beta = None
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'beta': beta
        }

#################################################################################################
# Function Name - generate_trading_signals
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function generates trading signals based on technical indicators
#################################################################################################
    

    def generate_trading_signals(self) -> tuple[str, list, list]:
        """Generate trading signals based on technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.calculate_technical_indicators()
        current = df.iloc[-1]
        prev = df.iloc[-2]

        signals = []
        score = 0
        predictTopgainer = []
        
        # RSI Signals
        if current['RSI'] < self.rsi_oversold:
            signals.append(f"Oversold (RSI: {current['RSI']:.2f})")
            score += 1
        elif current['RSI'] > self.rsi_overbought:
            signals.append(f"Overbought (RSI: {current['RSI']:.2f})")
            score -= 1
            
        # MACD Signals
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("MACD Bullish Crossover")
            score += 1
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("MACD Bearish Crossover")
            score -= 1
            
        # Moving Average Signals
        if current['Close'] > current['SMA_200']:
            signals.append("Price above 200 SMA (Bullish)")
            score += 0.5
        else:
            signals.append("Price below 200 SMA (Bearish)")
            score -= 0.5
            
        # Bollinger Bands Signals
        if current['Close'] < current['BB_Lower']:
            signals.append("Price below lower Bollinger Band (Potential Buy)")
            score += 1
        elif current['Close'] > current['BB_Upper']:
            signals.append("Price above upper Bollinger Band (Potential Sell)")
            score -= 1
        
                # Volume Signals
        if current['Volume'] > current['Volume_SMA_20'] * 1.5:
            if current['Close'] > current['Open']:
                signals.append("High Volume Bullish Day (Confirmation)")
                score += 1
            else:
                signals.append("High Volume Bearish Day (Confirmation)")
                score -= 1
        
        # Money Flow Index Signals
        if current['MFI'] < 20:
            signals.append(f"MFI Oversold ({current['MFI']:.2f})")
            score += 1
        elif current['MFI'] > 80:
            signals.append(f"MFI Overbought ({current['MFI']:.2f})")
            score -= 1

        # Chaikin Money Flow Signals
        if current['CMF'] > 0.1:
            signals.append("Positive Chaikin Money Flow (Bullish)")
            score += 0.5
        elif current['CMF'] < -0.1:
            signals.append("Negative Chaikin Money Flow (Bearish)")
            score -= 0.5
            

         # Price ROC Signals
        if current['Price_ROC'] > 5:
            signals.append(f"Strong Price Momentum (ROC: {current['Price_ROC']:.2f}%)")
            score += 0.5
        elif current['Price_ROC'] < -5:
            signals.append(f"Weak Price Momentum (ROC: {current['Price_ROC']:.2f}%)")
            score -= 0.5
            
            
        # Generate recommendation based on score
        if score >= 2:
            recommendation = "Strong Buy"
            predictTopgainer.append(score)
        elif score > 0:
            recommendation = "Buy"
            predictTopgainer.append(score)

        elif score == 0:
            recommendation = "Hold"
            predictTopgainer.append(score)

        elif score > -2:
            recommendation = "Sell" 
            # predictTopgainer.append(score)


        else:
            recommendation = "Strong Sell"
            
        return recommendation, signals, predictTopgainer
    
    def fetch_top_gainers(self) -> str:
            # get top gainer data
            # Define URLs
        nse_url = "https://www.nseindia.com"  # NSE homepage (to get cookies)
        api_url_topgainer = "https://www.nseindia.com/api/live-analysis-variations?index=gainers"
        api_url_toplooser = "https://www.nseindia.com/api/live-analysis-variations?index=loosers"


# Define headers
        headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    }

# Create a session
        session = requests.Session()
        self.recommendationTopGainer = {}

        try:
             # Step 1: Get cookies from the NSE homepage
            session.get(nse_url, headers=headers)

            # Step 2: Use the session with cookies to fetch API data of topgainer
            gainer_response = session.get(api_url_topgainer, headers=headers)
            gainer_response.raise_for_status()  # Raise an error if the request fails

            # Step 3: Use the session with cookies to fetch API data of topgainer
            looser_response = session.get(api_url_toplooser, headers=headers)
            looser_response.raise_for_status()  # Raise an error if the request fails

            # Step 4: Print the data
            top_gainer_data = gainer_response.json()
            top_looser_data = looser_response.json()

            top_gainer_symbols = [item["symbol"] for item in top_gainer_data["NIFTY"].get("data", [])]
            logger.info("Symbols extracting from top gainer api : %s", top_gainer_symbols)
            logger.error("Error in fetching symbol data")

            top_looser_symbols = [items["symbol"] for items in top_looser_data["NIFTY"].get("data", [])]
            logger.info("Symbols extracting from top looser api : %s", top_looser_symbols)
            
            symbols = top_gainer_symbols + top_looser_symbols
            print(symbols)
            for symbol in symbols:
                try:
                    logger.info("Symbol running in loop %s", symbol)
                    print(symbol)
                    self.fetch_stock_data(symbol)
                    self.calculate_technical_indicators()
                    _, _, score_list = self.generate_trading_signals()
                    if score_list:
                        # self.recommendationTopGainer.append(score_list[0])
                        self.recommendationTopGainer[symbol] = score_list[0]
                        print(score_list[0])
                except Exception as e :
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    self.recommendationTopGainer[symbol] = None
            return symbols
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top gainers top loosers: {str(e)}")
            return []

#################################################################################################
# Function Name - analyze_stock
# Date - 18th Mar 2025
# Description - This function provides a comprehensive analysis of a stock
#################################################################################################

    def analyze_stock(self, symbol: str, exchange: str = "NS", start_date: str = None) -> dict:
        """
        Perform a comprehensive analysis of a stock.
        
        Parameters:
        symbol (str): Stock symbol
        exchange (str): Exchange suffix (default: "NS" for NSE)
        start_date (str): Start date for analysis (default: 1 year ago)
        
        Returns:
        dict: Complete analysis results
        """
        # Fetch stock data
        self.fetch_stock_data(symbol, exchange, start_date)
        
        # Calculate technical indicators
        tech_data = self.calculate_technical_indicators()
        
        # Generate trading signals
        recommendation, signals, predictTopgainer = self.generate_trading_signals()
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics()
        
        # Identify support and resistance levels
        support_levels, resistance_levels = self.identify_support_resistance(tech_data)
        
        # Analyze price action
        price_action = self.analyze_price_action(tech_data)

        # New fundamental analysis
        fundamental_analysis = self.perform_fundamental_analysis(symbol)
        
        # Prepare chart data
        chart_data = []
        for date, row in tech_data.iterrows():
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'sma20': round(float(row['SMA_20']), 2) if pd.notnull(row['SMA_20']) else None,
                'sma50': round(float(row['SMA_50']), 2) if pd.notnull(row['SMA_50']) else None,
                'rsi': round(float(row['RSI']), 2) if pd.notnull(row['RSI']) else None,
                'macd': round(float(row['MACD']), 2) if pd.notnull(row['MACD']) else None,
                'signal': round(float(row['MACD_Signal']), 2) if pd.notnull(row['MACD_Signal']) else None,
                'volume': int(row['Volume']),
                'volumeSMA20': int(row['Volume_SMA_20']) if pd.notnull(row['Volume_SMA_20']) else None
            })
        
        # Create result dictionary
        result = {
            'symbol': symbol,
            'exchange': exchange,
            'recommendation': recommendation,
            'fundamentalAnalysis': fundamental_analysis,
            'currentPrice': round(float(self.stock_data['Close'][-1]), 2),
            'signals': signals,
            'predictTopgainer': predictTopgainer,
            'riskMetrics': {
                'sharpeRatio': round(float(risk_metrics['sharpe_ratio']), 2),
                'volatility': round(float(risk_metrics['volatility']), 2),
                'maxDrawdown': round(float(risk_metrics['max_drawdown']), 2),
                'beta': round(float(risk_metrics['beta']), 2) if risk_metrics['beta'] else None
            },
            'supportResistance': {
                'support': [round(float(level), 2) for level in support_levels],
                'resistance': [round(float(level), 2) for level in resistance_levels]
            },
            'priceAction': {
                'trend': price_action['trend'],
                'patterns': price_action['patterns'],
                'momentum': price_action['momentum'],
                'volumeSignal': price_action['volume_signal'],
                'recentVolatility': price_action['recent_volatility']
            },
            'chartData': chart_data
        }
        
        return result


#################################################################################################
# Function Name - analyze_top_gainers
# Date - 18th Mar 2025
# Description - This function analyzes the top gainers stocks with enhanced analysis
#################################################################################################

    def analyze_top_gainers(self, limit: int = 5) -> list:
        """
        Fetch and analyze top gainers stocks with enhanced analysis.
        
        Parameters:
        limit (int): Number of top gainers to analyze
        
        Returns:
        list: List of analysis results for each top gainer
        """
        try:
            # Fetch top gainers
            top_gainers = self.fetch_top_gainers(limit)
            
            # Analyze each stock
            results = []
            
            for gainer in top_gainers:
                symbol = gainer['symbol']
                logger.info(f"Analyzing top gainer: {symbol}")
                
                # Skip non-equity series if the data comes from NSE API
                if 'series' in gainer and gainer['series'] != 'EQ':
                    continue
                    
                try:
                    # Perform comprehensive analysis
                    analysis = self.analyze_stock(symbol)
                    
                    # Add percent change from the gainer data
                    analysis['percentChange'] = gainer['percentChange']
                    
                    results.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing top gainers: {str(e)}")
            raise


#################################################################################################
# Function Name - index
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function renders the index.html template
#################################################################################################

@app.route('/')
def index():
    return render_template('index.html')

#################################################################################################
# Function Name - analyze
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function analyzes the stock data and returns the analysis results
#################################################################################################

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        exchange = data.get('exchange', 'NS')
        start_date = data.get('startDate')

        analyzer = IndianStockAnalyzer()
        analysis = analyzer.analyze_stock(symbol, exchange, start_date)

        response = {
            'success': True,
            'data': analysis
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@click.command()
@click.option('--limit', default=5, help='Number of top gainers to analyze')
def analyze_top_gainers_cli(limit):
    """Command line function to analyze top gainers and print results."""
    try:
        print(f"Fetching and analyzing top {limit} gainers from NSE...")
        
        analyzer = IndianStockAnalyzer()
        results = analyzer.analyze_top_gainers(limit)
        
        print("\n=================== TOP GAINERS ANALYSIS ===================\n")
        
        for idx, result in enumerate(results, 1):
            print(f"#{idx}: {result['symbol']} - ₹{result['currentPrice']} ({result['percentChange']}%)")
            print(f"Recommendation: {result['recommendation']}")
            
            print("Price Action:")
            print(f"  - Trend: {result['priceAction']['trend']}")
            if result['priceAction']['patterns']:
                print(f"  - Patterns: {', '.join(result['priceAction']['patterns'])}")
            print(f"  - Momentum: {result['priceAction']['momentum']}")
            if result['priceAction']['volumeSignal']:
                print(f"  - Volume: {result['priceAction']['volumeSignal']}")
            
            print("Support & Resistance:")
            print(f"  - Support Levels: {', '.join([f'₹{level}' for level in result['supportResistance']['support']])}")
            print(f"  - Resistance Levels: {', '.join([f'₹{level}' for level in result['supportResistance']['resistance']])}")
            
            print("Signals:")
            for signal in result['signals']:
                print(f"  - {signal}")
                
            print("Risk Metrics:")
            print(f"  - Sharpe Ratio: {result['riskMetrics']['sharpeRatio']}")
            print(f"  - Volatility: {result['riskMetrics']['volatility']}%")
            print(f"  - Max Drawdown: {result['riskMetrics']['maxDrawdown']}%")
            if result['riskMetrics']['beta']:
                print(f"  - Beta: {result['riskMetrics']['beta']}")
            print("\n" + "-" * 60 + "\n")
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

# updated