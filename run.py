import pandas as pd
import plotly.express as px
import datetime
from test import predictnow

def graph(ticker):
    df = pd.read_csv('./csvs/' + '{}_final'.format(ticker) + 'NEW.csv')
    
    date = df['date']
    date = date.tolist()
    for i in range(15):
        date.append(date[-1] + 86400)
    
    close = df['close']
    predict = df['predictclose_15']
    
    close = pd.Series(close.tolist() + [0]*15)
    predict = pd.Series( ([0]*15 + predict.tolist()))
    
    date = pd.Series(date).apply(datetime.datetime.fromtimestamp)
    
    predict_df = pd.DataFrame(dict(date = date[15:], Cost = predict[15:]))
    predict_df['type'] = pd.Series(["Predicted"]*len(date))
    
    close_df = pd.DataFrame(dict(date = date[:len(date)-15], Cost = close[:len(date)-15]))
    close_df['type'] = pd.Series(["Actual"]*len(date))
    
    final = pd.concat([predict_df, close_df])
    
    fig = px.line(final, x='date', y="Cost", color = "type")
    fig.show()
    fig.write_html("graphs/" + '{}_comp.html'.format(ticker))
    
print("\nWe have graphs for the following companies:")
print("\n1. Apple\n2. Cisco\n3. Facebook\n4. Sony\n5. Google")

while True:
    user_input = input("What company would you like us to help you with:")
    user_input.lower()

    if user_input in 'Apple' or "APPL": 
        predictnow("Apple")
        graph("Apple")
        print("\n\nCheck out your graph at: ./graph/Apple_comp.html\n\n")
    elif user_input in 'Cisco' or 'CSCO':
        graph("Cisco")
        print("\n\nCheck out your graph at: ./graph/Cisco_comp.html\n\n")
    elif user_input in 'Metaverse' or 'FB' or 'Facebook':
        predictnow("FB")
        graph("FB")
        print("\n\nCheck out your graph at: ./graph/FB_comp.html\n\n")
    elif user_input in 'Sony':
        predictnow("Sony")
        graph("Sony")
        print("\n\nCheck out your graph at: ./graph/Sony_comp.html\n\n")
    elif user_input in 'Google' or 'GOOGL':
        predictnow("Google")
        graph("Google")
        print("\n\nCheck out your graph at: ./graph/Google_comp.html\n\n")
    else:
        print("Sorry we don't have that data on that stock yet.")