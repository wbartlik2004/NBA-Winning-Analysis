import requests
import pandas as pd
from datetime import datetime

ODDS_API_KEY = "9b116d394a3e85ee6a4cf2afd557da69"

def calculate_implied_probability(american_odds):
    """Convert American odds to implied probability"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def get_nba_odds_multi_source():
    """Get NBA odds from DraftKings, Kalshi, and Polymarket"""
    url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/'
    
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us,us_ex',
        'markets': 'h2h',
        'oddsFormat': 'american'  # This still applies to sportsbooks only
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        
        print("="*80)
        print("API USAGE")
        print("="*80)
        print(f"Requests used: {used}")
        print(f"Requests remaining: {remaining}")
        print("="*80 + "\n")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Fetch data
print("Fetching NBA odds...\n")
odds_data = get_nba_odds_multi_source()

if odds_data:
    print(f"✅ Received data for {len(odds_data)} games\n")
    
    results = []
    
    for game in odds_data:
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        commence_time = game.get('commence_time')
        
        # Convert time
        try:
            dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %I:%M %p ET')
        except:
            time_str = commence_time
        
        bookmakers = game.get('bookmakers', [])
        
        draftkings_odds = None
        draftkings_prob = None
        kalshi_prob = None
        polymarket_prob = None
        
        for book in bookmakers:
            book_key = book.get('key', '')
            markets = book.get('markets', [])
            
            if not markets:
                continue
                
            h2h_market = markets[0]
            outcomes = h2h_market.get('outcomes', [])
            
            # Find home team value
            home_value = None
            for outcome in outcomes:
                if outcome['name'] == home_team:
                    home_value = outcome['price']
                    break
            
            if home_value is None:
                continue
            
            # Different handling based on source
            if book_key == 'draftkings':
                draftkings_odds = home_value
                # Convert American odds to probability
                draftkings_prob = calculate_implied_probability(home_value)
                
            elif book_key == 'kalshi':
                # Kalshi gives DECIMAL probability (0-1)
                # If it's between 0 and 1, it's already a probability
                if 0 <= home_value <= 1:
                    kalshi_prob = home_value
                else:
                    # If it's in price format ($0.65), divide by 100
                    kalshi_prob = home_value / 100 if home_value > 1 else home_value
                    
            elif book_key == 'polymarket':
                # Polymarket gives DECIMAL probability (0-1)
                if 0 <= home_value <= 1:
                    polymarket_prob = home_value
                else:
                    # If it's in price format ($0.65), divide by 100
                    polymarket_prob = home_value / 100 if home_value > 1 else home_value
        
        results.append({
            'Commence_Time': time_str,
            'Home_Team': home_team,
            'Away_Team': away_team,
            'DraftKings_Odds': draftkings_odds if draftkings_odds else 'N/A',
            'DraftKings_Home_Prob': f"{draftkings_prob:.1%}" if draftkings_prob else 'N/A',
            'Kalshi_Home_Prob': f"{kalshi_prob:.1%}" if kalshi_prob else 'N/A',
            'Polymarket_Home_Prob': f"{polymarket_prob:.1%}" if polymarket_prob else 'N/A',
        })
    
    if results:
        df = pd.DataFrame(results)
        
        print("="*80)
        print("NBA GAMES TODAY - ALL SOURCES")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Save to Excel
        excel_file = "nba_odds_all_sources_today.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"\n✅ Saved to: {excel_file}")
        
        # Coverage
        dk_count = (df['DraftKings_Odds'] != 'N/A').sum()
        kalshi_count = (df['Kalshi_Home_Prob'] != 'N/A').sum()
        poly_count = (df['Polymarket_Home_Prob'] != 'N/A').sum()
        
        print("\n" + "="*80)
        print("COVERAGE")
        print("="*80)
        print(f"Total games: {len(df)}")
        print(f"DraftKings: {dk_count} games ({dk_count/len(df)*100:.0f}%)")
        print(f"Kalshi: {kalshi_count} games ({kalshi_count/len(df)*100:.0f}%)")
        print(f"Polymarket: {poly_count} games ({poly_count/len(df)*100:.0f}%)")
        print("="*80)
        
        print("\n" + "="*80)
        print("KEY DIFFERENCES")
        print("="*80)
        print("• DraftKings: Sportsbook - has ALL games")
        print("• Kalshi: Prediction market - only select games")
        print("• Polymarket: Prediction market - only select games")
        print("• DraftKings prob includes vig (>100% total)")
        print("• Kalshi/Polymarket are 'true' probabilities")
        print("="*80)
