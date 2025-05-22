import dash
from dash import dcc, html, Input, Output
import joblib
import numpy as np
import datetime


# Load model
model = joblib.load("/Users/trdny_josephjungermann/Documents/the_real_deal/20250515_DFW_ML_HousingPrices/DFW_house_price_model.pkl")

all_counties = ["Collin", "Dallas", "Denton", "Ellis", "Hood", "Hunt",
                "Johnson", "Kaufman", "Parker", "Rockwall", "Somervell",
                "Tarrant", "Wise"]

# for year feature dropdown
current_year = datetime.datetime.now().year

app = dash.Dash(__name__)
server = app.server  # For deployment

app.layout = html.Div([
    html.H2("Dallas-Fort Worth Home Price Predictor"),

    html.Label("Bedrooms"),
    dcc.Slider(1, 5, 1, value=3, id="beds", marks={i: str(i) for i in range(1, 6)}),

    html.Label("Bathrooms"),
    dcc.Slider(1, 4, 0.5, value=2, id="baths", marks={i: str(i) for i in range(1, 5)}),

    html.Label("Square Footage"),
    dcc.Slider(2000, 8000, 2000, value=4000, id="sqft",
               marks={i: str(i) for i in range(2000, 8001, 2000)}),

    html.Label("Year Built"),
    dcc.Dropdown(
        options=[{"label": year, "value": year} for year in range(1920, current_year + 1)],
        value=2025,
        id="yearBuilt"
    ),
    
    html.Label("County"),
    dcc.Dropdown(
        options=[{"label": c, "value": c} for c in all_counties],
        value="Dallas",
        id="county"
    ),

    html.Br(),
    html.Div(id="prediction-output", style={"fontSize": "24px", "marginTop": "20px"})
])

@app.callback(
    Output("prediction-output", "children"),
    Input("beds", "value"),
    Input("baths", "value"),
    Input("sqft", "value"),
    Input("yearBuilt", "value"),
    Input("county", "value")
)
def predict_price(beds, baths, sqft, yearBuilt, county):
    county_features = [1 if county == c else 0 for c in all_counties]
    X = np.array([[beds, baths, sqft, yearBuilt] + county_features])
    price = model.predict(X)[0]
    return f"Predicted Price: ${price:,.0f}"

if __name__ == "__main__":
    app.run(debug=True)