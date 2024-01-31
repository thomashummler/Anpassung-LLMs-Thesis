
import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import json
from openai import OpenAI

#openai API Key aus Umgebungsvariable lesen
API_KEY = os.environ["API_KEY"]
openai_api_key = API_KEY

#Pfad zu Database
file_path = 'DATA.xlsx'

#Einlesen der Database in pandas Dataframe
Database = pd.read_excel(file_path)

# Seed für Sub Classing setzen
seed_value = 42
np.random.seed(seed_value)

#Sublassing druchführen
df_groupByColor_Rieker = Database.groupby('Main_Color', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByShoeType_Rieker = Database.groupby('main_category', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByGender_Rieker = Database.groupby('Warengruppe', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupBySaison_Rieker = Database.groupby('Saison_Catch', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByMaterial_Rieker = Database.groupby('EAS Material', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))

#Subclassing zusammenführen und Duplikate eliminieren
result_df = pd.concat([df_groupByColor_Rieker, df_groupByShoeType_Rieker, df_groupByGender_Rieker, df_groupBySaison_Rieker, df_groupByMaterial_Rieker], ignore_index=True)
result_df = result_df.drop_duplicates(subset='ID', keep='first')
Database = result_df


# Der gegebene String
def extractJsonOutOfResponse(string):
  text = string
  # Regex, um das JSON-Objekt zu finden

  json_str = re.search(r'```json\n(.+?)\n```', text, re.DOTALL)

  if json_str:
      # Konvertieren des gefundenen Strings in ein JSON-Objekt
      json_obj = json.loads(json_str.group(1))
     
      return json_obj
  else:
      return "NO JSOn"

#Methode um Informationen aus JSON zu filtern und Datenbankfilterung durch Aufruf der Methode   filter_rieker_database durchzuführen
def setDataAndFilterWithJSON(json_string):

  data = json.loads(json_string)

  data_color = data["Color"]
  data_shoe_type = data.get("Shoe Type")
  data_gender = data.get("Gender")
  data_season = data.get("Season")
  data_material = data.get("Material")



  return filter_rieker_database(color=data_color, shoe_type=data_shoe_type, gender=data_gender, season=data_season, material=data_material )

#Datenbankfilterungs Methode die vor jeder Filterung auch gecheckt ob die Eingabe überhaupt gesetzt ist
def filter_rieker_database(color, shoe_type, gender, season, material):
    conditions = []

    if color != "":
        conditions.append(Database["Main_Color"] == color)

    if shoe_type != "":
        conditions.append(Database["main_category"] == shoe_type)

    if gender != "":
        conditions.append(Database["Warengruppe"] == gender)

    if season != "":
        conditions.append(Database["Saison_Catch"] == season)

    if material != "":
        conditions.append(Database["EAS Material"] == material)

 #   if closure != "":
 #       conditions.append(Rieker_Database["Verschluss"] == closure)

    if conditions:
        filtered_df = Database.loc[pd.concat(conditions, axis=1).all(axis=1)]
        return filtered_df
    else:
        return Database



# Agent der die Usereingabe verstehen soll und als JSON zurückliefert
if 'chatVerlauf_Information' not in st.session_state:
    st.session_state.chatVerlauf_Information = []
    st.session_state.chatVerlauf_Information=[{
    "role": "system",
    "content": f"You try to recognize various information from the user and return it in json format. "
               f"You will get a {{jsondata}} json File. If it is None it doesent have to btoher"
               f"This json Data File contains the Information you got out of the User Inputs in previous interactions"
               f"The following information should be recognized: Color, shoe type, gender, season, material "
               f"The color must correspond to one of the following colors: schwarz, grau, braun, beige, weiß, rot, blau, "
               f"grün, gelb, gold, lila, mehrfarbig, rosa, bunt, metallisch, silber, orange. The shoe type must correspond "
               f"to one of the following: Slipper, Chelsea Boots, Biker Boots, Stiefel, Stiefeletten, Sneaker, Halbschuhe, "
               f"Rieker EVOLUTION, Pantoletten, Sandalen, Sandaletten. Gender must be either Damen oder Herren. "
               f"Season must be one of the following: Winter, Sommer. Material must match one of the "
               f"following: Glattleder, Lackleder, Rauhleder/Stretch, Glattleder/Stretch, Rauleder, Kunstleder, Kunstlack, "
               f"Textil, Reptilleder, Kunststoff. TIf the user's input does not match the given information on color, "
               f"shoe type, gender, season, material, select the most suitable information unless the user has not entered "
               f"anything for the corresponding information. Then return a "" for the respective information and not null. The JSON format "
               f"should be: {{'Color': 'User Specified Color', 'Shoe Type': 'User Specified Shoe Type', "
               f"'Gender': 'User Specified Gender', 'Season': 'User Specified Season', "
               f"'Material': 'User Specified Material'}} In the following interaction a interactions with the User some Filers may already be set. "
               f"Recognize that some filters may already are set in {{jsondata}} and just change them if u got new Informations from the User about this Filters"
    }]


client = OpenAI(
    api_key= API_KEY
)

#Zusätzlicher Agent für Begrüßung 
chatVerlauf_UserInteraction=[{
        "role": "system",
           "content": f"You are a polite and helpful assistant who should help the user find the right shoes out of a Shoes Database.That's why you greet the user first and ask how you can help them. Right this welcoming Message in German.  "
        }]
chat_User = client.chat.completions.create(
         model="gpt-4-1106-preview",
         messages=chatVerlauf_UserInteraction
        )
start_Message_System = chat_User.choices[0].message.content




st.title("Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": start_Message_System})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Annahmed es User Inputs -> Dadurch wird Retrieval Prozess initiert 
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

    user_input = prompt
    #User Input an Agent übergebend er JSON generiert
    st.session_state.chatVerlauf_Information.append({"role": "user", "content": user_input}) # User Input in Chatverlauf hinzufügen bevor API Call über Agent gemacht wird
    chat_Filter = client.chat.completions.create(    #API Call für Agent machen der über NLP die Eingaben des Users interpretiert und den vordefinierten Variabeln zuweist und diese als JSON zurückliefert
    model="gpt-4-1106-preview", 
    response_format={ "type": "json_object" },
    messages= st.session_state.chatVerlauf_Information
    )
    jsondata =  chat_Filter.choices[0].message.content #JSON zwischenspeichern
    st.session_state.chatVerlauf_Information.append({"role": "assistant", "content": jsondata})           #JSON Datei in Chatverlauf für Agent der JSON Generierung hinzufügen
    print(jsondata)
    filtered_DF = setDataAndFilterWithJSON(jsondata) #Dataframe nach JSON Informationen filtern und in filtered_DF zwischenspeichern
    print(filtered_DF)
    #print(filtered_DF)
    lengthOfFilteredDatabase = filtered_DF.shape[0] #Länge des Dataframes und somit die Anzahl der Schuhe zwischenspeichern
    #print(lengthOfFilteredDatabase)

    # Falls zweiter Agent noch nicht lief, diesen initieren und an diesen JSON, das gefilterte Dataframe und die Länge des Dataframes weitergeben
    if 'chatVerlauf_UserInteraction' not in st.session_state:
        st.session_state.chatVerlauf_UserInteraction = []
        st.session_state.chatVerlauf_UserInteraction.append({
        "role": "system",
        "content": f"You are a polite and helpful assistant who should help the user find the right shoesv out of a database." 
                   f"You should always answer in the Language that the User is using."  
                   f"If the User is trying is asking about something else than soes then explain to him that your purpose is it to find the right shoes. "
                   f"After every User Input the RAG Pipeline is getting started again ist retrieving to you a new List of new Documents"
                   f"You get a JSON file {jsondata} with the following variables Color, Shoe Type, Gender, Season and Material."
                   f" These are filters with which you want to help the user to find the right shoe for the customer." 
                   f" If a variable could not yet be recognized from the User_Input, there is a '' in the JSON file." 
                   f" If this is the case, explicitly ask the user again for this filters."
                   f" You also have to mention how many shoes are already filtered. The current amount is {lengthOfFilteredDatabase} "
                   f" <= 5 them give a deailted descripton about each shoe in {filtered_DF}."
                   f" These are the currently filtered shoe : {filtered_DF}." 
                   f" Please describe the shoes in a continuous text and not in embroidery dots. " 
                   f" If tells u to give to him the currently best fitting shoes, choose two shoes out of {filtered_DF} that are best fitting to the User Input" 
    })   
    st.session_state.chatVerlauf_UserInteraction.append({"role": "user", "content": user_input}) #User Input an Agent weitergeben
    chat_User = client.chat.completions.create( #Agent antwort generieren lassen
    model="gpt-4-1106-preview",
    messages=st.session_state.chatVerlauf_UserInteraction
    )
    system_Message = chat_User.choices[0].message.content
    st.session_state.chatVerlauf_UserInteraction.append({"role": "assistant", "content": system_Message}) # Antwort des Agent in Historie des Agent speichern
    full_response = system_Message
    message_placeholder.markdown(full_response) 
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})# Antwort im Chat anzeigen lassen
    print(st.session_state.chatVerlauf_UserInteraction)
    






