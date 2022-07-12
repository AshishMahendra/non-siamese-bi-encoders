# flake8: noqa

import json
# import requests
# import numpy as np
# all_label = ["searchplace", "getplacedetails", "bookrestaurant", "gettrafficinformation","compareplaces", "sharecurrentlocation", "requestride", "getdirections", "shareeta", "getweather"]
# test_data = ["What is the cheapest restaurant between Balthazar and Lombardi's?", 'Which restaurant is the cheapest, Shake Shack or Five Guys ?', "What is the cheapest restaurant between Balthazar and Lombardi's?", "What is the cheapest restaurant between Balthazar and Lombardi's?", "What is the cheapest restaurant between Balthazar and Lombardi's?", "What is the most expensive restaurant between Per Se and Rao's?", 'Order a cab at Shake Shack to go to the Empire State Building', 'Order a cab at Shake Shack to go to the Empire State Building', 'Get a taxi to the nearest japanese restaurant', 'Is there any Uber around?', "Get a Lyft car at Scott's place", "Get a Lyft car at Scott's place", "I need the weather at Jo's place around 8 pm", "What's the weather like at work?", 'What will the weather be like when I land in Paris?', "What's the weather like at my parents'?", "What's the weather like near my upcoming event?", 'Is it going to rain tomorrow?', 'Find me a good expensive japanese restaurant near work', 'Find a good fried chicken restaurant that is not a fast food', 'Find me the closest theatre for tonight', 'Find a good fried chicken restaurant that is not a fast food', 'Find me the finest sushi restaurant in the area of my next meeting', 'Find me a good expensive japanese restaurant near work', "Is there a children's menu at The Standard Grill?", 'Are there some tips to know when going to Battery Park?', 'How crowded is Per Se right now?', "When is Rao's the most crowded?", 'How crowded is the bar near my place?',
#              'Is there valet parking at Kang Ho Dong Baekjeong?', 'Share my current location', "Send my current location to the friends I'm meeting with", 'Share my current location', 'Share my current location', "Send my current location to the friends I'm meeting with", "Send my current location to the friends I'm meeting with", "How's the traffic from here to Central Park?", "How's the traffic from here to Central Park?", 'Is the road to work congested?', 'Is there any traffic on US 20?', "How's the traffic from here to Central Park?", "How's the traffic from here to Central Park?", 'Book me a table for 8:45pm at a restaurant with wifi near my Airbnb', "Book a table for 6 people at Mr Donahue's for tomorrow's lunch", 'I want a table in a good japanese restaurant near Trump tower', 'Book a table for four people at Mondrian Soho for 8pm', 'Find me an outdoor table at Five leaves for 3', "Book a table for today's lunch at Eggy's Diner for 3 people", 'Show me the way to go to work', 'I want to go to Boston with the quickest itinerary', 'Directions to JFK airport at 7am', 'Cycling directions to my surf lesson at 10am', 'Transit directions to Barcelona Wine Bar', 'Driving directions to Tavern on the Green', 'Send my ETA to the guests of my apartment', "Send my ETA to the girl I'm supposed to have dinner with", "Send my ETA to the girl I'm supposed to have dinner with", 'Send a message to Montgomery with my arrival time', 'Send my ETA to the guests of my apartment', 'Send a message to my boss with my ETA']
# test_label = ['bookrestaurant', 'bookrestaurant', 'bookrestaurant', 'bookrestaurant', 'bookrestaurant', 'bookrestaurant', 'requestride', 'requestride', 'requestride', 'requestride', 'requestride', 'requestride', 'getplacedetails', 'getplacedetails', 'getplacedetails', 'getplacedetails', 'getplacedetails', 'getplacedetails', 'getdirections', 'getdirections', 'getdirections', 'getdirections', 'getdirections', 'getdirections', 'compareplaces', 'compareplaces', 'compareplaces', 'compareplaces', 'compareplaces', 'compareplaces', 'sharecurrentlocation',
#               'sharecurrentlocation', 'sharecurrentlocation', 'sharecurrentlocation', 'sharecurrentlocation', 'sharecurrentlocation', 'searchplace', 'searchplace', 'searchplace', 'searchplace', 'searchplace', 'searchplace', 'shareeta', 'shareeta', 'shareeta', 'shareeta', 'shareeta', 'shareeta', 'getweather', 'getweather', 'getweather', 'getweather', 'getweather', 'getweather', 'gettrafficinformation', 'gettrafficinformation', 'gettrafficinformation', 'gettrafficinformation', 'gettrafficinformation', 'gettrafficinformation']
# counter = 0
# print("Inference started ..")
# for data, lbl in zip(test_data, test_label):
#     test_infer_request = {
#         "contexts": [data],
#         "candidates": all_label
#     }
#     r = requests.post('http://127.0.0.1:8000/infer', json=test_infer_request)
#     if lbl == r.json():
#         counter += 1
# print(f"accuracy_score  : {counter/(len(test_label))}")
# # response = requests.post(
# #     "http://localhost:8000/get_context_emb",
# #     json={"contexts": test_data}
# # )
# # print(len(response.json()))
# con_data = []
# for data in test_data:
#     test_infer_request = {
#         "contexts": [data]}
#     response = requests.post(
#         "http://localhost:8000/get_context_emb",
#         json=test_infer_request
#     )
#     con_data.append(response.json())
# lbl_data = []
# for lbl in all_label:
#     test_infer_request = {
#         "candidates": [lbl]}
#     response = requests.post(
#         "http://localhost:8000/get_candidate_emb",
#         json=test_infer_request
#     )
#     lbl_data.append(response.json())
# score_dat = []
# counter = 0
# hit = 0
# print("staring infer now ;;;;;;;;")
# for data in con_data:
#     score_dat = []
#     for lbl in lbl_data:
#         test_infer_request = {
#             "vec_a": data,
#             "vec_b": lbl,
#             "meta": "string"
#         }
#         response = requests.post(
#             "http://localhost:8000/cosine_sim",
#             json=test_infer_request
#         )
#         score_dat.append(response.json())
#     if (all_label[np.argmax(score_dat)] == test_label[counter]):
#         hit += 1
#     counter = +1

# print(f"total hit  : {hit}")
pred_list = [
    "gettrafficinformation",
    "gettrafficinformation",
    "gettrafficinformation",
    "sharecurrentlocation",
    "gettrafficinformation",
    "gettrafficinformation",
    "bookrestaurant",
    "bookrestaurant",
    "gettrafficinformation",
    "bookrestaurant",
    "gettrafficinformation",
    "gettrafficinformation",
    "bookrestaurant",
    "gettrafficinformation",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "gettrafficinformation",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "gettrafficinformation",
    "gettrafficinformation",
    "bookrestaurant",
    "bookrestaurant",
    "gettrafficinformation",
    "gettrafficinformation",
    "gettrafficinformation",
    "gettrafficinformation",
    "gettrafficinformation",
    "gettrafficinformation",
    "gettrafficinformation",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "bookrestaurant",
    "gettrafficinformation",
    "gettrafficinformation",
    "bookrestaurant",
    "gettrafficinformation",
    "gettrafficinformation",
    "bookrestaurant",
    "bookrestaurant",
    "gettrafficinformation",
    "gettrafficinformation"
]
with open("testing.txt", "r") as jsonfile:
    t_data = json.load(jsonfile)
print(len(t_data["candidates"]))
print(len(pred_list))
counter = 0
for i, j in zip(t_data["candidates"], pred_list):
    if i == j:
        counter += 1
print(counter)
