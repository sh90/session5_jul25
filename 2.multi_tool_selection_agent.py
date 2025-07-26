import streamlit as st
import openai
import requests
from openai import OpenAI
import data_info

# what's the weather in chicago
# --- API Keys ---
openai.api_key = data_info.open_ai_key
RAPIDAPI_KEY = '766ea387a6msh84ff46479ed9b4bp18b495jsna547d73e1ef5'
IMDB_API_URL = "https://imdb236.p.rapidapi.com/api/imdb/autocomplete"
AVIATIONSTACK_API_KEY = "56420807c490bd835b6e922d2b983fbb"
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"

# --- Helper API Calls ---
def call_flight_status_api(flight_number):
    params = {'access_key': AVIATIONSTACK_API_KEY, 'flight_iata': flight_number}
    response = requests.get(AVIATIONSTACK_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('data'):
            flight_info = data['data'][0]
            status = flight_info['flight_status']
            departure = flight_info['departure']['airport']
            arrival = flight_info['arrival']['airport']
            return f"Flight {flight_number} is currently {status}. Departure: {departure}, Arrival: {arrival}."
        else:
            return "Sorry, couldn't find information for that flight."
    return "Sorry, couldn't retrieve flight data."

def call_imdb_api(movie_title):
    headers = {
        'x-rapidapi-host': 'imdb236.p.rapidapi.com',
        'x-rapidapi-key': RAPIDAPI_KEY
    }
    params = {'query': movie_title}
    response = requests.get(IMDB_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return "Sorry, couldn't retrieve movie information."

def call_exchange_api(base_currency, target_currency):
    url = f"https://open.er-api.com/v6/latest/{base_currency.upper()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('result') == 'success':
            rate = data['rates'].get(target_currency.upper())
            if rate:
                return f"1 {base_currency.upper()} = {rate} {target_currency.upper()}."
            return f"Sorry, couldn't find exchange rate for {target_currency}."
    return "Sorry, couldn't access the exchange rate service."

# --- Main Agent ---
def gpt4o_mini_super_agent(user_input):
    tool_selection_prompt = f"""You are a multi-tool assistant who can use external APIs:

    - Weather API (for weather questions)
    - Movie API (for movie questions)
    - Currency Exchange API (for currency conversion)
    - Flight Tracker API (for flight status)

    When the user asks:
    - About the weather -> "CALL_WEATHER_API: <city-name>"
    - About a movie -> "CALL_MOVIE_API: <movie-title>"
    - About currency -> "CALL_EXCHANGE_API: <base-currency> to <target-currency>"
    - About a flight -> "CALL_FLIGHT_API: <flight-number>"

    Otherwise, do not answer and provide options related to tools you have access to.

    User input: "{user_input}"
    """

    client = OpenAI(api_key=data_info.open_ai_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=tool_selection_prompt,
        temperature=0,
    )
    assistant_reply = response.output_text

    # Process the reply
    if assistant_reply.startswith("CALL_WEATHER_API:"):
        city_name = assistant_reply.split(":", 1)[1].strip()
        # Geocode
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        lat = lon = None
        r = requests.get(geo_url, params={'name': city_name, 'count': 1})
        if r.status_code == 200 and r.json().get('results'):
            result = r.json()['results'][0]
            lat = result['latitude']
            lon = result['longitude']
        if lat and lon:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            r = requests.get(weather_url)
            if r.status_code == 200:
                weather = r.json().get('current_weather', {})
                final_prompt = f"The weather in {city_name} is: {weather}. Please create a friendly response."
            else:
                return "Sorry, weather data couldn't be fetched."
        else:
            return f"Sorry, couldn't locate {city_name}."

    elif assistant_reply.startswith("CALL_MOVIE_API:"):
        movie_title = assistant_reply.split(":", 1)[1].strip()
        api_result = call_imdb_api(movie_title)
        final_prompt = f"The movie information about {movie_title} is: {api_result}. Please create a friendly response."

    elif assistant_reply.startswith("CALL_EXCHANGE_API:"):
        parts = assistant_reply.split(":", 1)[1].strip().split(" to ")
        if len(parts) == 2:
            base_currency, target_currency = parts
            api_result = call_exchange_api(base_currency, target_currency)
            final_prompt = f"The exchange rate information is: {api_result}. Please create a friendly response."
        else:
            return "Sorry, I couldn't understand the currency conversion request."

    elif assistant_reply.startswith("CALL_FLIGHT_API:"):
        flight_number = assistant_reply.split(":", 1)[1].strip()
        api_result = call_flight_status_api(flight_number)
        final_prompt = f"The flight status information is: {api_result}. Please create a friendly response."

    else:
        return assistant_reply  # Non-tool use fallback

    client = OpenAI(api_key=data_info.open_ai_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=final_prompt,
        temperature=0,
    )
    return response.output_text

# --- Streamlit UI ---
st.set_page_config(page_title="Super Agent", page_icon="====")
st.title("=====Multi-Tool Super Agent=====")
st.markdown("Ask me about **weather**, **movies**, **currency exchange**, or **flight status**.")

user_query = st.text_input("Ask your question:")
if st.button("Submit") or user_query:
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = gpt4o_mini_super_agent(user_query)
        st.markdown("### 🤖 Answer")
        st.write(answer)
