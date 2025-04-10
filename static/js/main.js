/////////////////////////////////////////////////////////////////////////////////////////////////////////
// File Name - main.js
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This file contains the main logic for the frontend of the application.
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fucntion Name - formatCurrency
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to format the currency in the form of ₹.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function formatCurrency(value) {
    let modified
    if(value>=0 && value<=100000){
        return `₹${parseFloat(value).toFixed(2)}`;
    }
    else if(value>=100000 && value<=10000000){
        console.log(value)
         modified = value/100000
        console.log("modified",modified)
         return `₹${parseFloat(modified).toFixed(2)} Lac`;
    }
    else if(value>=10000000){
        console.log(value)
        modified = value/10000000
        console.log("modified",modified)
        return `₹${parseFloat(modified).toFixed(2)} Cr`;
    }
    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fucntion Name - formatPercentage
// Author - Ojas Ulhas Dighe
// Date - 28th Mar 2025
// Description - This function is used to format the percentage in the form of 0.00%.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function formatPercentage(value) {
    return `${parseFloat(value).toFixed(2)}%`;
}


// //////////////////////////////////////////////////////////////
// current date and time format
// Update immediately 
function getFormattedDateTime() {
    const date = new Date();
    
    // Get date components
    const day = String(date.getDate()).padStart(2, '0');
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = monthNames[date.getMonth()];
    const year = String(date.getFullYear()).slice(-2);
    
    // Get time components
    let hours = date.getHours();
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    
    // Convert to 12-hour format
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
    hours = String(hours).padStart(2, '0');

    return `${day} ${month} ${year}, ${hours}:${minutes} ${ampm}`;
}

// ////////////////////////////////////////////////////////////////////
// update recomendation colour
function updateRecomendationColor(recomendation){
    const recDiv = document.getElementById('recDiv')
    if(recomendation === "Strong Buy" || recomendation === "Buy"){
        recDiv.style.background="#30a166"
    }
    else if(recomendation === "Hold" ){
        recDiv.style.background="orange"
    }
    else{
        recDiv.style.background="#f43f5e"
    }
}



// ////////////////////////////////////////////////////////////////////////
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    card.classList.toggle('expanded');
    
    const chevron = card.querySelector('.chevron-icon');
    chevron.classList.toggle('chevron-up');
}

// Ensure the fundamentals card is expanded on load
document.addEventListener('DOMContentLoaded', function() {
    const fundamentalsCard = document.getElementById('fundamentals-card');
    fundamentalsCard.classList.add('expanded');
    const chevron = fundamentalsCard.querySelector('.chevron-icon');
    chevron.classList.add('chevron-up');
});


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - updateFundamentalAnalysisSection
// Author - Ojas Ulhas Dighe
// Date - 28th Mar 2025
// Description - This function updates the fundamental analysis section of the UI
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function updateFundamentalAnalysisSection(fundamentalData) {
    // Basic Info Section
    document.getElementById('companyName').textContent = fundamentalData.basic_info.company_name || 'N/A';
    document.getElementById('sector').textContent = fundamentalData.basic_info.sector || 'N/A';
    document.getElementById('industry').textContent = fundamentalData.basic_info.industry || 'N/A';

    // Valuation Metrics
    document.getElementById('marketCap').textContent = formatCurrency(fundamentalData.valuation_metrics.market_cap);
    document.getElementById('peRatio').textContent = fundamentalData.valuation_metrics.pe_ratio.toFixed(2);
    document.getElementById('forwardPE').textContent = fundamentalData.valuation_metrics.forward_pe.toFixed(2);
    document.getElementById('priceToBook').textContent = fundamentalData.valuation_metrics.price_to_book.toFixed(2);
    document.getElementById('dividendYield').textContent = formatPercentage(fundamentalData.valuation_metrics.dividend_yield);

    // Financial Health
    document.getElementById('totalRevenue').textContent = formatCurrency(fundamentalData.financial_health.total_revenue);
    document.getElementById('grossProfit').textContent = formatCurrency(fundamentalData.financial_health.gross_profit);
    document.getElementById('netIncome').textContent = formatCurrency(fundamentalData.financial_health.net_income);
    document.getElementById('totalDebt').textContent = formatCurrency(fundamentalData.financial_health.total_debt);
    document.getElementById('debtToEquity').textContent = fundamentalData.financial_health.debt_to_equity.toFixed(2);
    document.getElementById('returnOnEquity').textContent = formatPercentage(fundamentalData.financial_health.return_on_equity);

    // Growth Metrics
    document.getElementById('revenueGrowth').textContent = formatPercentage(fundamentalData.growth_metrics.revenue_growth);
    document.getElementById('earningsGrowth').textContent = formatPercentage(fundamentalData.growth_metrics.earnings_growth);
    document.getElementById('profitMargins').textContent = formatPercentage(fundamentalData.growth_metrics.profit_margins);

    // Fundamental Recommendation
    const fundamentalRecommendationBadge = document.getElementById('fundamentalRecommendationBadge');
    fundamentalRecommendationBadge.textContent = fundamentalData.recommendation;
    fundamentalRecommendationBadge.className = `recommendation-badge ${getRecommendationClass(fundamentalData.recommendation)}`;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - getRecommendationClass
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to get the recommendation class based on the recommendation.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function getRecommendationClass(recommendation) {
    const classes = {
        'Strong Buy': 'strong-buy',
        'Buy': 'buy',
        'Hold': 'hold',
        'Sell': 'sell',
        'Strong Sell': 'strong-sell'
    };
    return classes[recommendation] || '';
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createPriceChart(chartData) {
    const dates = chartData.map(d => d.date);
    const prices = chartData.map(d => d.price);
    const sma20 = chartData.map(d => d.sma20);
    const sma50 = chartData.map(d => d.sma50);

    const traces = [
        {
            name: 'Price',
            x: dates,
            y: prices,
            type: 'scatter',
            line: { color: '#2563eb' }
        },
        {
            name: '20 SMA',
            x: dates,
            y: sma20,
            type: 'scatter',
            line: { color: '#16a34a' }
        },
        {
            name: '50 SMA',
            x: dates,
            y: sma50,
            type: 'scatter',
            line: { color: '#dc2626' }
        }
    ];

    const layout = {
        title: 'Price Action',
        autosize: true,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (₹)' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('priceChart', traces, layout);
}  

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createPriceChart(chartData) {
    const dates = chartData.map(d => d.date);
    const prices = chartData.map(d => d.price);
    const sma20 = chartData.map(d => d.sma20);
    const sma50 = chartData.map(d => d.sma50);

    const traces = [
        {
            name: 'Price',
            x: dates,
            y: prices,
            type: 'scatter',
            line: { color: '#2563eb' }
        },
        {
            name: '20 SMA',
            x: dates,
            y: sma20,
            type: 'scatter',
            line: { color: '#16a34a' }
        },
        {
            name: '50 SMA',
            x: dates,
            y: sma50,
            type: 'scatter',
            line: { color: '#dc2626' }
        }
    ];

    const layout = {
        paper_bgcolor: '#121218',   // Background of the entire chart
        plot_bgcolor: '#121218', 
        font: {
            color: 'white'  // Adjust text color for readability
          },
        title: 'Price Action',
        autosize: true,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (₹)' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('priceChart', traces, layout);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createIndicatorsChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the indicators chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createIndicatorsChart(chartData) {
    const dates = chartData.map(d => d.date);
    const rsi = chartData.map(d => d.rsi);
    const macd = chartData.map(d => d.macd);
    const signal = chartData.map(d => d.signal);

    const traces = [
        {
            name: 'RSI',
            x: dates,
            y: rsi,
            type: 'scatter',
            yaxis: 'y1',
            line: { color: '#8b5cf6' }
        },
        {
            name: 'MACD',
            x: dates,
            y: macd,
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#3b82f6' }
        },
        {
            name: 'Signal',
            x: dates,
            y: signal,
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#ef4444' }
        }
    ];

    const layout = {
        title: 'Technical Indicators',
        autosize: true,
        paper_bgcolor: '#121218',   // Background of the entire chart
        plot_bgcolor: '#121218', 
        font: {
            color: 'white'  // Adjust text color for readability
          },
        xaxis: { title: 'Date' },
        yaxis: { 
            title: 'RSI',
            domain: [0.6, 1]
        },
        yaxis2: {
            title: 'MACD',
            domain: [0, 0.4]
        },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 },
        grid: { rows: 2, columns: 1, pattern: 'independent' }
    };

    Plotly.newPlot('indicatorsChart', traces, layout);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - analyzeStock
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to analyze the stock.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

async function analyzeStock() {
    const stockSymbol = document.getElementById('stockSymbol').value;
    const exchange = document.getElementById('exchange').value;
    const startDate = document.getElementById('startDate').value;


    if (!stockSymbol) {
        showError('Please enter a stock symbol');
        return;
    }

    // Show loading state
    document.getElementById('loadingIndicator').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('errorMessage').classList.add('hidden');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: stockSymbol,
                exchange: exchange,
                startDate: startDate
            })
        });

        const result = await response.json();
        console.log(result)
        if (!result.success) {
            throw new Error(result.error);
        }

        if(result.data.predictTopgainer.length===0){
            document.getElementById('topRecomendation').style.display = "none"
            
        }

        updateUI(result.data);
        document.getElementById('tickerName').innerText = stockSymbol.toUpperCase()
    } catch (error) {
        showError(error.message || 'An error occurred while analyzing the stock');
    } finally {
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - updateUI
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to update the UI.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function updateUI(data) {
    // Show results container
    document.getElementById('results').classList.remove('hidden');
    // show charts
    document.getElementById('charts').style.display = 'grid'
//  UPDATE TIME
    document.querySelector('.date').textContent = getFormattedDateTime();
    // Update the color of the recomendation div
    updateRecomendationColor(data.recommendation)
    // update logo of stocks
    document.getElementById('img').setAttribute('src', data.fundamentalAnalysis.basic_info.logoURL)

    // Update current price and recommendation
    document.getElementById('currentPrice').textContent = formatCurrency(data.currentPrice);
    const recommendationBadge = document.getElementById('recommendationBadge');
    recommendationBadge.textContent = data.recommendation;
    // recommendationBadge.className = `recommendation-badge ${getRecommendationClass(data.recommendation)}`;

    // Update signals list
    const signalsList = document.getElementById('signalsList');
    signalsList.innerHTML = '';
    data.signals.forEach(signal => {
        const li = document.createElement('li');
        li.textContent = signal;
        signalsList.appendChild(li);
        li.setAttribute('class', "analysis-text")
    });

    // Update risk metrics
    const metrics = data.riskMetrics;
    document.getElementById('sharpeRatio').textContent = metrics.sharpeRatio.toFixed(2);
    document.getElementById('volatility').textContent = metrics.volatility.toFixed(2) + '%';
    document.getElementById('maxDrawdown').textContent = metrics.maxDrawdown.toFixed(2) + '%';
    document.getElementById('beta').textContent = metrics.beta ? metrics.beta.toFixed(2) : '-';
    // document.getElementById('alpha').textContent = metrics.alpha ? metrics.alpha.toFixed(2) : '-';
    // document.getElementById('correlation').textContent = metrics.correlation ? metrics.correlation.toFixed(2) : '-';

        // Added Fundamental Analysis Section
        if (data.fundamentalAnalysis) {
            updateFundamentalAnalysisSection(data.fundamentalAnalysis);
            document.getElementById('fundamentalAnalysisSection').classList.remove('hidden');
        } else {
            document.getElementById('fundamentalAnalysisSection').classList.add('hidden');
        }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

    createPriceChart(data.chartData);
    createIndicatorsChart(data.chartData);

    // update new stock recomendation
    // Function to populate the object

    const div = document.getElementById('otherStockRecomendationList');
    
    // Clear existing list items
    div.innerHTML = '';
    
    // Create and append new list items
    for (const [stock, score] of Object.entries(data.predictTopgainer)) {
        const li = document.createElement('h4');
        const divElement = document.createElement('div');
        divElement.appendChild(li)
        li.textContent = `${stock}`;
        div.appendChild(divElement);
        li.setAttribute('class', 'list-group-item')
        divElement.setAttribute('class', 'list-group-item-div');
    }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - showError
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to show the error message.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    popup()
    // Set default date to one year ago
    const defaultDate = new Date();
    defaultDate.setFullYear(defaultDate.getFullYear() - 1);
    document.getElementById('startDate').value = defaultDate.toISOString().split('T')[0];

    // Add enter key listener for stock symbol input
    document.getElementById('stockSymbol').addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            analyzeStock();
        }
    });
});

// Add window resize handler for charts
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        const results = document.getElementById('results');
        if (!results.classList.contains('hidden')) {
            Plotly.relayout('priceChart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            });
            Plotly.relayout('indicatorsChart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true,
                'yaxis2.autorange': true
            });
        }
    }, 250);

    
});
 
// update disclamour functionality 

function popup() {
     // Disclaimer functionality
     const disclaimerToggle = document.getElementById('disclaimerToggle');
     const disclaimerPopup = document.getElementById('disclaimerPopup');
     const closeDisclaimer = document.getElementById('closeDisclaimer');
     
     
     let disclaimerTimeout;
 
     

     function showDisclaimer() {
        //  disclaimerPopup.classList.remove('hidden');
         disclaimerPopup.style.display = "flex";

         disclaimerTimeout = setTimeout(() => {
             disclaimerPopup.style.display = "none"
         }, 3000); // 3 seconds
     }
 
     function hideDisclaimer() {
        //  disclaimerPopup.classList.add('hidden');
         disclaimerPopup.style.display = "none";

         clearTimeout(disclaimerTimeout);
     }
 
     disclaimerToggle.addEventListener('click', showDisclaimer);
     closeDisclaimer.addEventListener('click', hideDisclaimer);
 
     // Optional: Close when clicking outside popup
     disclaimerPopup.addEventListener('click', (e) => {
        console.log("click")
         if (e.target === disclaimerPopup) {
             hideDisclaimer();
         }
     });
}